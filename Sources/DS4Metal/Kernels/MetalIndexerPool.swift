import Foundation
import Metal

// Phase 9 / Stage A5: indexer per-head score collapse + softmax pooling. Faithful
// ports dispatching the real metal/dsv4_misc.metal kernels
// kernel_dsv4_indexer_weighted_sum and kernel_dsv4_softmax_pool. Strides chosen
// row-major; the kernels are generic over strides (validates the dispatch).

extension MetalRuntime {
    /// Indexer relevance score per compressed row (decode, 1 token):
    /// score[row] = scale * sum_head max(q[head]·k_row, 0) * weights[head].
    /// Dispatches the real kernel_dsv4_indexer_score_one_direct (n_head=64, head_dim=128).
    /// q: [nHead*headDim] f32, weights: [nHead] f32, indexComp: [nComp*headDim] f32.
    public func indexerScoreOne(q: [Float], weights: [Float], indexComp: [Float],
                                nComp: Int, nHead: Int = 64, headDim: Int = 128, scale: Float) throws -> [Float] {
        precondition(q.count >= nHead * headDim && weights.count >= nHead && indexComp.count >= nComp * headDim)
        let args = Self.indexerScoreArgs(nComp: nComp, nHead: nHead, headDim: headDim, scale: scale)
        guard let qbuf = device.makeBuffer(bytes: q, length: q.count * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(bytes: weights, length: weights.count * 4, options: .storageModeShared),
              let cbuf = device.makeBuffer(bytes: indexComp, length: indexComp.count * 4, options: .storageModeShared),
              let sbuf = device.makeBuffer(length: nComp * 4, options: .storageModeShared) else { throw MetalError.bufferAlloc }
        let pso = try pipeline("kernel_dsv4_indexer_score_one_direct")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(qbuf, offset: 0, index: 1)
        enc.setBuffer(wbuf, offset: 0, index: 2)
        enc.setBuffer(cbuf, offset: 0, index: 3)
        enc.setBuffer(sbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength((128 + 4) * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: nComp, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = sbuf.contents().bindMemory(to: Float.self, capacity: nComp)
        return Array(UnsafeBufferPointer(start: p, count: nComp))
    }

    /// 72-byte ds4_metal_args_dsv4_indexer_scores_fused (single token decode).
    static func indexerScoreArgs(nComp: Int, nHead: Int, headDim: Int, scale: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 72)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        u32(0, UInt32(nComp)); u32(4, 1); u32(8, UInt32(nHead)); u32(12, UInt32(headDim)); u32(16, 0); u32(20, 4)
        u64(24, UInt64(nHead * headDim * 4))   // q_token_stride
        u64(32, UInt64(headDim * 4))           // q_head_stride
        u64(40, UInt64(nHead * 4))             // weights_token_stride
        u64(48, UInt64(headDim * 4))           // index_row_stride
        u64(56, UInt64(nComp * 4))             // score_token_stride
        f32(64, scale)
        return b
    }

    /// Collapse per-head indexer scores into one score per (token, comp):
    /// out[it][ic] = sum_ih max(scores[ih][it][ic], 0) * weights[it][ih] * scale.
    /// scores: nHead x nTokens x nComp, weights: nTokens x nHead.
    public func indexerWeightedSum(scores: [Float], weights: [Float],
                                   nHead: Int, nTokens: Int, nComp: Int, scale: Float) throws -> [Float] {
        precondition(scores.count >= nHead * nTokens * nComp && weights.count >= nTokens * nHead)
        let args = Self.indexerWeightedSumArgs(nHead: nHead, nTokens: nTokens, nComp: nComp, scale: scale)
        guard let sbuf = device.makeBuffer(bytes: scores, length: scores.count * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(bytes: weights, length: weights.count * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nTokens * nComp * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_indexer_weighted_sum")
        let total = nComp * nTokens
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(wbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, max(1, total)), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = dbuf.contents().bindMemory(to: Float.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    /// Softmax-weighted pooling over `nRows` compressed rows for one pool:
    /// out[d] = sum_r softmax(score[r])[r] * kv[r][d], for d in 0..<width.
    /// kv: nRows x width, score: nRows.
    public func softmaxPool(kv: [Float], score: [Float], nRows: Int, width: Int) throws -> [Float] {
        precondition(kv.count >= nRows * width && score.count >= nRows)
        let args = Self.softmaxPoolArgs(nRows: nRows, width: width)
        guard let kvbuf = device.makeBuffer(bytes: kv, length: kv.count * 4, options: .storageModeShared),
              let scbuf = device.makeBuffer(bytes: score, length: score.count * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: width * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_softmax_pool")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(kvbuf, offset: 0, index: 1)
        enc.setBuffer(scbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, max(1, width)), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = dbuf.contents().bindMemory(to: Float.self, capacity: width)
        return Array(UnsafeBufferPointer(start: p, count: width))
    }

    /// 120-byte ds4_metal_args_dsv4_indexer_weighted_sum (14 fields + float).
    static func indexerWeightedSumArgs(nHead: Int, nTokens: Int, nComp: Int, scale: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 120)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let compBytes = UInt64(nComp) * 4
        i64(0, Int64(nComp)); i64(8, Int64(nTokens)); i64(16, Int64(nHead))   // ne00, ne01, ne02
        u64(24, 4); u64(32, compBytes); u64(40, compBytes * UInt64(nTokens))  // nb00(ic), nb01(it), nb02(ih)
        i64(48, Int64(nHead)); i64(56, Int64(nTokens))                        // ne10, ne11
        u64(64, 4); u64(72, UInt64(nHead) * 4)                                // nb10(ih), nb11(it)
        i64(80, Int64(nComp)); i64(88, Int64(nTokens))                        // ne0, ne1
        u64(96, 4); u64(104, compBytes)                                       // nb0(ic), nb1(it)
        f32(112, scale)                                                       // scale
        return b
    }

    /// 104-byte ds4_metal_args_dsv4_softmax_pool. Single pool: ne0=width, ne1=1.
    static func softmaxPoolArgs(nRows: Int, width: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 104)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let widthBytes = UInt64(width) * 4
        i64(0, Int64(nRows)); i64(8, 1); i64(16, 1)                  // ne00(rows), ne01, ne02
        u64(24, widthBytes); u64(32, 4); u64(40, 0)                  // nb00(ir, kv row), nb01(id=d), nb02(ic)
        u64(48, 4); u64(56, 0); u64(64, 0)                          // nb10(ir, score), nb11(id), nb12(ic)
        i64(72, Int64(width)); i64(80, 1)                           // ne0(width=d), ne1(ic)
        u64(88, 4); u64(96, 0)                                      // nb0(id), nb1(ic)
        return b
    }
}
