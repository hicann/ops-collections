/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "hyperloglog_ref.h"
#include "detail/hyperloglog/hash.h"
#include "simt_api/device_sync_functions.h"

namespace aclco::detail::hyperloglog {
constexpr std::uint32_t threads = 256;
constexpr std::uint32_t bins = 65;
constexpr AscendC::SyncAllConfig mte3ToMte2Sync = {PIPE_MTE3, PIPE_MTE2};
COLLECTION_SIMT_DEVICE std::uint64_t ThreadIndex()
{
    return static_cast<std::uint64_t>(AscendC::Simt::GetBlockIdx()) * threads + AscendC::Simt::GetThreadIdx();
}
COLLECTION_SIMT_DEVICE std::uint64_t ThreadStride()
{
    return static_cast<std::uint64_t>(AscendC::Simt::GetBlockNum()) * threads;
}
COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void ClearSimt(__gm__ std::uint32_t* words, std::uint32_t count)
{
    for (auto i = ThreadIndex(); i < count; i += ThreadStride())
        words[i] = 0;
}
template <int = 0>
COLLECTION_AIV_GLOBAL void ClearKernel(__gm__ std::uint32_t* words, std::uint32_t count)
{
    AscendC::Simt::VF_CALL<ClearSimt>(AscendC::Simt::Dim3{threads}, words, count);
}
template <class Key>
COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void AddSimt(__gm__ std::uint32_t* words, __gm__ Key* keys,
                                                             std::uint64_t count, std::uint32_t precision)
{
    HyperLogLogRef<Key> ref(words, precision);
    for (auto i = ThreadIndex(); i < count; i += ThreadStride()) {
        Key const key = keys[i];
        ref.Add(key);
    }
}
template <class Key>
COLLECTION_AIV_GLOBAL void AddKernel(__gm__ std::uint32_t* words, __gm__ Key* keys, std::uint64_t count,
                                     std::uint32_t precision)
{
    AscendC::Simt::VF_CALL<AddSimt<Key>>(AscendC::Simt::Dim3{threads}, words, keys, count, precision);
}
// Each rank is at most 52. Setting each byte's high bit prevents cross-byte
// borrows; its subtraction result selects the larger of the two original bytes.
COLLECTION_SIMT_DEVICE std::uint32_t MaxPackedRanks(std::uint32_t a, std::uint32_t b)
{
    auto const high = ((a | 0x80808080U) - b) & 0x80808080U;
    auto const mask = (high >> 7U) * 255U;
    return (a & mask) | (b & ~mask);
}
COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void MergeSimt(__gm__ std::uint32_t* words,
                                                               __gm__ std::uint32_t const* other, std::uint32_t count)
{
    for (auto i = ThreadIndex(); i < count; i += ThreadStride()) {
        words[i] = MaxPackedRanks(words[i], other[i]);
    }
}
template <int = 0>
COLLECTION_AIV_GLOBAL void MergeKernel(__gm__ std::uint32_t* words, __gm__ std::uint32_t const* other,
                                       std::uint32_t count)
{
    AscendC::Simt::VF_CALL<MergeSimt>(AscendC::Simt::Dim3{threads}, words, other, count);
}
COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void HistogramSimt(__gm__ std::uint32_t const* words,
                                                                   __gm__ std::uint32_t* output,
                                                                   __ubuf__ std::uint32_t* counts,
                                                                   std::uint32_t wordCount)
{
    auto const lane = AscendC::Simt::GetThreadIdx();
    if (lane < bins)
        counts[lane] = 0;
    asc_syncthreads();
    for (auto i = ThreadIndex(); i < wordCount; i += ThreadStride()) {
        auto const word = words[i];
        for (std::uint32_t shift = 0; shift < 32; shift += 8) {
            asc_atomic_add(counts + ((word >> shift) & 255U), 1U);
        }
    }
    asc_syncthreads();
    if (lane < bins)
        output[AscendC::Simt::GetBlockIdx() * bins + lane] = counts[lane];
}
template <int = 0>
COLLECTION_AIV_GLOBAL void HistogramKernel(__gm__ std::uint32_t const* words, __gm__ std::uint32_t* output,
                                           std::uint32_t wordCount)
{
    __ubuf__ std::uint32_t counts[96];
    AscendC::Simt::VF_CALL<HistogramSimt>(AscendC::Simt::Dim3{threads}, words, output, counts, wordCount);
}

// One private sketch per core avoids fine-grained GM atomics during the input scan.
// Five 6-bit ranks per word allow a 256 KiB public sketch to fit in dynamic UB.
constexpr std::uint32_t privateThreads = 1024;
template <unsigned PackedRanks>
COLLECTION_HOST_DEVICE constexpr unsigned PackedRankBits()
{
    static_assert(PackedRanks == 1 || PackedRanks == 2 || PackedRanks == 4 || PackedRanks == 5,
                  "Unsupported HyperLogLog rank packing");
    if constexpr (PackedRanks == 1)
        return 32;
    if constexpr (PackedRanks == 2)
        return 16;
    if constexpr (PackedRanks == 4)
        return 8;
    return 6;
}
template <unsigned Precision>
COLLECTION_HOST_DEVICE constexpr unsigned TiledInputBytes()
{
    if constexpr (Precision <= 14)
        return 64 * 1024;
    if constexpr (Precision == 15)
        return 44 * 1024;
    if constexpr (Precision == 16)
        return 76 * 1024;
    return 32 * 1024;
}
template <unsigned Precision>
static __attribute__((noinline)) __simt_callee__[aicore] std::uint32_t RankFromFinalHashLow(HashWord state,
                                                                                            std::uint32_t hashHigh)
{
    auto const low = FinalHashLow(state, hashHigh);
    return low == 0 ? 65U - Precision : 33U - Precision + static_cast<unsigned>(__clz(static_cast<int>(low)));
}
template <class Key, unsigned PackedRanks, unsigned Precision>
COLLECTION_SIMT_DEVICE void UpdateLocal(Key key, __ubuf__ std::uint32_t* local)
{
    constexpr unsigned precision = Precision;
    constexpr unsigned bits = PackedRankBits<PackedRanks>();
    constexpr unsigned mask = 0xffffffffU >> (32U - bits);
    auto const hashState = HashKeyBeforeFinalMultiply(key);
    auto const hashHigh = FinalHashHigh(hashState);
    auto const index = hashHigh >> (32U - precision);
    auto const hi = hashHigh & (0xffffffffU >> precision);
    auto rank = static_cast<std::uint32_t>(__clz(static_cast<int>(hi | 1U))) + 1U - precision;
    if (hi == 0)
        rank = RankFromFinalHashLow<Precision>(hashState, hashHigh);
    if constexpr (PackedRanks == 4) {
        // Every field only increases, so a byte load can safely reject a redundant update.
        if (reinterpret_cast<__ubuf__ std::uint8_t*>(local)[index] >= rank)
            return;
    }
    auto const shift = (index % PackedRanks) * bits;
    auto* word = local + index / PackedRanks;
    if constexpr (PackedRanks == 1) {
        if (*word < rank)
            asc_atomic_max(word, rank);
    } else {
        auto observed = *word;
        while (((observed >> shift) & mask) < rank) {
            auto const desired = (observed & ~(mask << shift)) | (rank << shift);
            auto const previous = asc_atomic_cas(word, observed, desired);
            if (previous == observed)
                break;
            observed = previous;
        }
    }
}
template <class Key, unsigned PackedRanks, unsigned Precision>
COLLECTION_SIMT_VF LAUNCH_BOUND(privateThreads) inline void PrivateAddSimt(__gm__ Key const* keys,
                                                                           __gm__ std::uint32_t* sketches,
                                                                           __ubuf__ std::uint32_t* local,
                                                                           std::uint32_t count)
{
    constexpr unsigned precision = Precision;
    constexpr unsigned bits = PackedRankBits<PackedRanks>();
    constexpr unsigned mask = 0xffffffffU >> (32U - bits);
    auto const lane = AscendC::Simt::GetThreadIdx();
    auto const m = 1U << precision;
    auto const localWords = (m + PackedRanks - 1) / PackedRanks;
    for (std::uint32_t i = lane; i < localWords; i += privateThreads)
        local[i] = 0;
    asc_syncthreads();
    auto const stride = static_cast<std::uint32_t>(AscendC::Simt::GetBlockNum()) * privateThreads;
    for (std::uint32_t i = static_cast<std::uint32_t>(AscendC::Simt::GetBlockIdx()) * privateThreads + lane; i < count;
         i += stride) {
        Key const key = keys[i];
        UpdateLocal<Key, PackedRanks, Precision>(key, local);
    }
    asc_syncthreads();
    auto* output = sketches + static_cast<std::uint64_t>(AscendC::Simt::GetBlockIdx()) * (m / 4);
    for (std::uint32_t i = lane; i < m / 4; i += privateThreads) {
        std::uint32_t word = 0;
        for (unsigned j = 0; j < 4; ++j) {
            auto const index = 4 * i + j;
            auto value = (local[index / PackedRanks] >> ((index % PackedRanks) * bits)) & mask;
            word |= value << (8 * j);
        }
        output[i] = word;
    }
}
template <class Key, unsigned PackedRanks, unsigned Precision>
COLLECTION_AIV_GLOBAL void PrivateAddKernel(__gm__ Key const* keys, __gm__ std::uint32_t* sketches, std::uint32_t count)
{
    auto* local = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
    AscendC::Simt::VF_CALL<PrivateAddSimt<Key, PackedRanks, Precision>>(AscendC::Simt::Dim3{privateThreads}, keys,
                                                                        sketches, local, count);
}

constexpr unsigned tiledThreads = 1024;
template <unsigned PackedRanks, unsigned Precision>
COLLECTION_SIMT_VF LAUNCH_BOUND(tiledThreads) inline void InitLocalSimt(__ubuf__ std::uint32_t* local)
{
    constexpr unsigned size = ((1U << Precision) + PackedRanks - 1) / PackedRanks;
    for (unsigned i = AscendC::Simt::GetThreadIdx(); i < size; i += tiledThreads)
        local[i] = 0;
}
template <unsigned Precision>
COLLECTION_SIMT_VF LAUNCH_BOUND(tiledThreads) inline void InitLocalFromPackedSimt(__ubuf__ std::uint32_t* local,
                                                                                  __gm__ std::uint32_t const* packed)
{
    constexpr unsigned words = (1U << Precision) / 4;
    for (unsigned i = AscendC::Simt::GetThreadIdx(); i < words; i += tiledThreads) {
        auto const word = packed[i];
        local[4 * i] = word & 0xffU;
        local[4 * i + 1] = (word >> 8) & 0xffU;
        local[4 * i + 2] = (word >> 16) & 0xffU;
        local[4 * i + 3] = word >> 24;
    }
}
template <class Key, unsigned PackedRanks, unsigned Precision>
COLLECTION_SIMT_VF LAUNCH_BOUND(tiledThreads) inline void AddTileSimt(__ubuf__ Key const* keys,
                                                                      __ubuf__ std::uint32_t* local, unsigned count)
{
    for (unsigned i = AscendC::Simt::GetThreadIdx(); i < count; i += tiledThreads) {
        Key const key = keys[i];
        UpdateLocal<Key, PackedRanks, Precision>(key, local);
    }
}
template <unsigned PackedRanks, unsigned Precision>
COLLECTION_SIMT_VF LAUNCH_BOUND(tiledThreads) inline void WriteLocalSimt(__ubuf__ std::uint32_t const* local,
                                                                         __gm__ std::uint32_t* sketches)
{
    constexpr unsigned bits = PackedRankBits<PackedRanks>();
    constexpr unsigned mask = 0xffffffffU >> (32U - bits);
    constexpr unsigned m = 1U << Precision;
    auto* output = sketches + static_cast<std::uint64_t>(AscendC::Simt::GetBlockIdx()) * (m / 4);
    for (unsigned i = AscendC::Simt::GetThreadIdx(); i < m / 4; i += tiledThreads) {
        unsigned word = 0;
        for (unsigned j = 0; j < 4; ++j) {
            unsigned index = 4 * i + j;
            word |= ((local[index / PackedRanks] >> ((index % PackedRanks) * bits)) & mask) << (8 * j);
        }
        output[i] = word;
    }
}
template <class Key>
__aicore__ inline void CopyTile(AscendC::TQue<AscendC::TPosition::VECIN, 2>& inputQueue,
                                AscendC::GlobalTensor<Key>& global, unsigned offset, unsigned end, unsigned tileSize)
{
    auto tensor = inputQueue.AllocTensor<Key>();
    unsigned size = end - offset < tileSize ? end - offset : tileSize;
    AscendC::DataCopyExtParams params{1, static_cast<std::uint32_t>(size * sizeof(Key)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<Key> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(tensor, global[offset], params, pad);
    inputQueue.EnQue(tensor);
}
template <unsigned PackedRanks, unsigned Precision>
__aicore__ inline AscendC::LocalTensor<std::uint32_t> PrepareTiledSketch(
    AscendC::TBuf<AscendC::TPosition::VECCALC>& sketchBuffer, AscendC::TQue<AscendC::TPosition::VECIN, 1>& initialQueue,
    __gm__ unsigned const* initialSketch)
{
    constexpr unsigned localBytes = (((1U << Precision) + PackedRanks - 1) / PackedRanks) * 4;
    AscendC::LocalTensor<std::uint32_t> sketchTensor;
    if constexpr (Precision == 16 || Precision == 17) {
        sketchTensor = initialQueue.AllocTensor<std::uint32_t>();
        if (initialSketch) {
            AscendC::GlobalTensor<unsigned> initialGlobal;
            initialGlobal.SetGlobalBuffer(const_cast<__gm__ unsigned*>(initialSketch), localBytes / 4);
            AscendC::DataCopyExtParams params{1, localBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<unsigned> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(sketchTensor, initialGlobal, params, pad);
        }
        initialQueue.EnQue(sketchTensor);
        sketchTensor = initialQueue.DeQue<std::uint32_t>();
    } else {
        sketchTensor = sketchBuffer.Get<std::uint32_t>();
    }
    auto* local = reinterpret_cast<__ubuf__ std::uint32_t*>(sketchTensor.GetPhyAddr());
    if constexpr (Precision == 15) {
        if (initialSketch) {
            AscendC::Simt::VF_CALL<InitLocalFromPackedSimt<Precision>>(AscendC::Simt::Dim3{tiledThreads}, local,
                                                                       initialSketch);
        } else {
            AscendC::Simt::VF_CALL<InitLocalSimt<PackedRanks, Precision>>(AscendC::Simt::Dim3{tiledThreads}, local);
        }
    } else if constexpr (Precision == 16 || Precision == 17) {
        if (!initialSketch)
            AscendC::Simt::VF_CALL<InitLocalSimt<PackedRanks, Precision>>(AscendC::Simt::Dim3{tiledThreads}, local);
    } else {
        AscendC::Simt::VF_CALL<InitLocalSimt<PackedRanks, Precision>>(AscendC::Simt::Dim3{tiledThreads}, local);
    }
    return sketchTensor;
}
template <class Key, unsigned PackedRanks, unsigned Precision>
__aicore__ inline void AddTiles(AscendC::TQue<AscendC::TPosition::VECIN, 2>& inputQueue,
                                AscendC::GlobalTensor<Key>& global, __ubuf__ std::uint32_t* local, unsigned begin,
                                unsigned end)
{
    constexpr unsigned tileSize = TiledInputBytes<Precision>() / sizeof(Key);
    if (begin < end)
        CopyTile<Key>(inputQueue, global, begin, end, tileSize);
    for (unsigned offset = begin; offset < end; offset += tileSize) {
        if (offset + tileSize < end)
            CopyTile<Key>(inputQueue, global, offset + tileSize, end, tileSize);
        auto tensor = inputQueue.DeQue<Key>();
        auto* input = reinterpret_cast<__ubuf__ Key*>(tensor.GetPhyAddr());
        unsigned size = end - offset < tileSize ? end - offset : tileSize;
        AscendC::Simt::VF_CALL<AddTileSimt<Key, PackedRanks, Precision>>(AscendC::Simt::Dim3{tiledThreads}, input,
                                                                         local, size);
        inputQueue.FreeTensor(tensor);
    }
}
template <class Key, unsigned PackedRanks, unsigned Precision>
COLLECTION_AIV_GLOBAL void TiledAddKernel(__gm__ Key const* keys, __gm__ std::uint32_t* sketches, unsigned count,
                                          __gm__ unsigned const* initialSketch)
{
    constexpr unsigned localBytes = (((1U << Precision) + PackedRanks - 1) / PackedRanks) * 4;
    constexpr unsigned tileBytes = TiledInputBytes<Precision>();
    constexpr unsigned tileSize = tileBytes / sizeof(Key);
    static_assert(localBytes + 2 * tileBytes <= 216 * 1024);
    static_assert(Precision != 15 || PackedRanks == 1, "Precision 15 initial sketches require one rank per local word");
    static_assert(Precision != 16 || PackedRanks == 4, "Precision 16 initial sketches require byte registers");
    static_assert(Precision != 17 || PackedRanks == 4, "Precision 17 initial sketches require byte registers");
    // The launch reserves this full UB range for SIMT access as well as DMA buffers.
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sketchBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> initialQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    if constexpr (Precision == 16 || Precision == 17)
        pipe.InitBuffer(initialQueue, 1, localBytes);
    else
        pipe.InitBuffer(sketchBuffer, localBytes);
    pipe.InitBuffer(inputQueue, 2, tileBytes);
    auto sketchTensor = PrepareTiledSketch<PackedRanks, Precision>(sketchBuffer, initialQueue, initialSketch);
    auto* local = reinterpret_cast<__ubuf__ std::uint32_t*>(sketchTensor.GetPhyAddr());
    AscendC::GlobalTensor<Key> global;
    global.SetGlobalBuffer(const_cast<__gm__ Key*>(keys), count);
    unsigned cores = AscendC::GetBlockNum();
    unsigned chunk = ((count + cores * tileSize - 1) / (cores * tileSize)) * tileSize;
    unsigned begin = AscendC::GetBlockIdx() * chunk;
    unsigned end = begin + chunk < count ? begin + chunk : count;
    AddTiles<Key, PackedRanks, Precision>(inputQueue, global, local, begin, end);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Simt::VF_CALL<WriteLocalSimt<PackedRanks, Precision>>(AscendC::Simt::Dim3{tiledThreads}, local, sketches);
    if constexpr (Precision == 16 || Precision == 17)
        initialQueue.FreeTensor(sketchTensor);
}

COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void ReduceSketchesSimt(__gm__ std::uint32_t* words,
                                                                        __ubuf__ std::uint32_t const* sketches,
                                                                        unsigned activeWords, unsigned sketchCount)
{
    for (unsigned i = AscendC::Simt::GetThreadIdx(); i < activeWords; i += threads) {
        auto result = words[i];
        for (unsigned c = 0; c < sketchCount; ++c)
            result = MaxPackedRanks(result, sketches[c * activeWords + i]);
        words[i] = result;
    }
}
template <int = 0>
COLLECTION_AIV_GLOBAL void ReduceSketchesKernel(__gm__ std::uint32_t* words, __gm__ std::uint32_t const* sketches,
                                                unsigned wordCount, unsigned sketchCount, unsigned tileWords)
{
    // Copy the same register interval from all private sketches in one strided DMA.
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
    pipe.InitBuffer(inputQueue, 1, tileWords * sketchCount * 4);
    AscendC::GlobalTensor<unsigned> global;
    global.SetGlobalBuffer(const_cast<__gm__ unsigned*>(sketches), wordCount * sketchCount);
    for (unsigned start = AscendC::GetBlockIdx() * tileWords; start < wordCount;
         start += AscendC::GetBlockNum() * tileWords) {
        unsigned size = wordCount - start < tileWords ? wordCount - start : tileWords;
        auto tensor = inputQueue.AllocTensor<unsigned>();
        AscendC::DataCopyExtParams params{static_cast<std::uint16_t>(sketchCount), size * 4, (wordCount - size) * 4, 0,
                                          0};
        AscendC::DataCopyPadExtParams<unsigned> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(tensor, global[start], params, pad);
        inputQueue.EnQue(tensor);
        tensor = inputQueue.DeQue<unsigned>();
        auto* local = reinterpret_cast<__ubuf__ unsigned*>(tensor.GetPhyAddr());
        AscendC::Simt::VF_CALL<ReduceSketchesSimt>(AscendC::Simt::Dim3{threads}, words + start, local, size,
                                                   sketchCount);
        inputQueue.FreeTensor(tensor);
    }
}

__aicore__ inline void ReduceP16Sketches(AscendC::TQue<AscendC::TPosition::VECIN, 2>& inputQueue,
                                         AscendC::LocalTensor<std::uint32_t>& sketchTensor,
                                         AscendC::GlobalTensor<std::uint32_t>& globalWords,
                                         AscendC::GlobalTensor<std::uint32_t>& globalSketches, unsigned reduceTileWords,
                                         AscendC::TEventID vectorToMte3, AscendC::TEventID mte2ToVector)
{
    constexpr unsigned wordCount = (1U << 16) / 4;
    unsigned const cores = AscendC::GetBlockNum();
    if (reduceTileWords == 0)
        return;
    // Rank bytes are independent, so Vector Max can reduce four packed registers
    // per uint32 word. Each active core loads only the public-sketch interval it owns.
    unsigned const reduceBlocks = (wordCount + reduceTileWords - 1) / reduceTileWords;
    if (AscendC::GetBlockIdx() < reduceBlocks) {
        unsigned const start = AscendC::GetBlockIdx() * reduceTileWords;
        unsigned const size = wordCount - start < reduceTileWords ? wordCount - start : reduceTileWords;
        AscendC::DataCopyExtParams currentParams{1, size * 4, 0, 0, 0};
        AscendC::DataCopyPadExtParams<std::uint32_t> currentPad{false, 0, 0, 0};
        AscendC::DataCopyPad(sketchTensor[start], globalWords[start], currentParams, currentPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVector);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVector);
        auto tensor = inputQueue.AllocTensor<std::uint32_t>();
        AscendC::DataCopyExtParams params{static_cast<std::uint16_t>(cores), size * 4, (wordCount - size) * 4, 0, 0};
        AscendC::DataCopyPadExtParams<std::uint32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(tensor, globalSketches[start], params, pad);
        inputQueue.EnQue(tensor);
        tensor = inputQueue.DeQue<std::uint32_t>();
        auto outputBytes = sketchTensor.ReinterpretCast<std::uint8_t>();
        auto inputBytes = tensor.ReinterpretCast<std::uint8_t>();
        unsigned const byteStart = start * sizeof(std::uint32_t);
        unsigned const byteCount = size * sizeof(std::uint32_t);
        for (unsigned core = 0; core < cores; ++core) {
            AscendC::Max(outputBytes[byteStart], outputBytes[byteStart], inputBytes[core * byteCount],
                         static_cast<std::int32_t>(byteCount));
        }
        inputQueue.FreeTensor(tensor);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
        AscendC::DataCopy(globalWords[start], sketchTensor[start], size);
    }
}
__aicore__ inline unsigned P16StageEnd(unsigned count, unsigned stageAlignment, unsigned target)
{
    if (stageAlignment == 0)
        return 0;
    unsigned const aligned = (target / stageAlignment) * stageAlignment;
    unsigned const boundary = aligned < stageAlignment ? stageAlignment : aligned;
    return count / 2 < boundary ? count / 2 : boundary;
}
__aicore__ inline void ReloadP16Sketch(AscendC::LocalTensor<std::uint32_t>& sketchTensor,
                                       AscendC::GlobalTensor<std::uint32_t>& globalWords,
                                       AscendC::TEventID mte2ToVector)
{
    AscendC::SyncAll<true, mte3ToMte2Sync>();
    AscendC::DataCopyExtParams params{1, 1U << 16, 0, 0, 0};
    AscendC::DataCopyPadExtParams<std::uint32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(sketchTensor, globalWords, params, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVector);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVector);
}
template <class Key>
COLLECTION_AIV_GLOBAL void FusedP16AddKernel(__gm__ Key const* keys, __gm__ std::uint32_t* words,
                                             __gm__ std::uint32_t* sketches, unsigned count)
{
    static_assert(std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::int64_t>);
    constexpr unsigned precision = 16;
    constexpr unsigned packedRanks = 4;
    constexpr unsigned wordCount = (1U << precision) / 4;
    constexpr unsigned localBytes = 1U << precision;
    constexpr unsigned tileBytes = TiledInputBytes<precision>();
    constexpr unsigned tileSize = tileBytes / sizeof(Key);
    static_assert(localBytes + 2 * tileBytes == 216 * 1024);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sketchBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    pipe.InitBuffer(sketchBuffer, localBytes);
    pipe.InitBuffer(inputQueue, 2, tileBytes);
    auto sketchTensor = sketchBuffer.Get<std::uint32_t>();
    auto* local = reinterpret_cast<__ubuf__ std::uint32_t*>(sketchTensor.GetPhyAddr());
    auto const vectorToMte3 = pipe.FetchEventID(AscendC::HardEvent::V_MTE3);
    auto const mte2ToVector = pipe.FetchEventID(AscendC::HardEvent::MTE2_V);
    AscendC::Simt::VF_CALL<InitLocalSimt<packedRanks, precision>>(AscendC::Simt::Dim3{tiledThreads}, local);

    AscendC::GlobalTensor<Key> globalKeys;
    globalKeys.SetGlobalBuffer(const_cast<__gm__ Key*>(keys), count);
    AscendC::GlobalTensor<std::uint32_t> globalWords;
    globalWords.SetGlobalBuffer(words, wordCount);
    AscendC::GlobalTensor<std::uint32_t> globalSketches;
    unsigned const cores = AscendC::GetBlockNum();
    globalSketches.SetGlobalBuffer(sketches, wordCount * cores);
    unsigned const reduceTileWords = (tileBytes / (cores * sizeof(std::uint32_t))) & ~7U;

    unsigned const stageAlignment = cores * tileSize;
    unsigned const stageEnds[]{P16StageEnd(count, stageAlignment, 1U << 20),
                               P16StageEnd(count, stageAlignment, 8U << 20), count};
    unsigned stageBegin = 0;

    for (unsigned stage = 0; stage < 3; ++stage) {
        unsigned const stageEnd = stageEnds[stage];
        unsigned const stageCount = stageEnd - stageBegin;
        unsigned const chunk = ((stageCount + cores * tileSize - 1) / (cores * tileSize)) * tileSize;
        unsigned const begin = stageBegin + AscendC::GetBlockIdx() * chunk;
        unsigned const end = begin + chunk < stageEnd ? begin + chunk : stageEnd;
        AddTiles<Key, packedRanks, precision>(inputQueue, globalKeys, local, begin, end);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
        AscendC::DataCopy(globalSketches[AscendC::GetBlockIdx() * wordCount], sketchTensor, wordCount);
        AscendC::SyncAll<true, mte3ToMte2Sync>();

        ReduceP16Sketches(inputQueue, sketchTensor, globalWords, globalSketches, reduceTileWords, vectorToMte3,
                          mte2ToVector);
        if (stage + 1 < 3)
            ReloadP16Sketch(sketchTensor, globalWords, mte2ToVector);
        stageBegin = stageEnd;
    }
}
} // namespace aclco::detail::hyperloglog
