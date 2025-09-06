//! Performance regression documentation and alternative optimization attempts.
//!
//! This file documents optimization approaches that were tried but caused performance
//! regressions or other issues, so we don't repeat the same mistakes.

/// SIMD Optimization Attempt - REVERTED due to performance regression
/// 
/// ## What was tried:
/// - Used Rust's `portable_simd` API to process 16 bytes simultaneously
/// - Implemented parallel character classification using SIMD masks
/// - Used bitmask operations and `trailing_zeros()` for boundary detection
/// - Added support for all valid identifier characters including `$`
/// 
/// ## Technical approach:
/// ```rust,ignore
/// fn simd_identifier_mask(chunk: &[u8; 16]) -> u16 {
///     let bytes = u8x16::from_array(*chunk);
///     let lower_alpha = bytes.simd_ge(u8x16::splat(b'a')) & bytes.simd_le(u8x16::splat(b'z'));
///     let upper_alpha = bytes.simd_ge(u8x16::splat(b'A')) & bytes.simd_le(u8x16::splat(b'Z'));
///     let digits = bytes.simd_ge(u8x16::splat(b'0')) & bytes.simd_le(u8x16::splat(b'9'));
///     let underscores = bytes.simd_eq(u8x16::splat(b'_'));
///     let dollar_signs = bytes.simd_eq(u8x16::splat(b'$'));
///     let id_mask = lower_alpha | upper_alpha | digits | underscores | dollar_signs;
///     id_mask.to_bitmask() as u16
/// }
/// ```
/// 
/// ## Why it was reverted:
/// - Performance regression detected in benchmarks
/// - SIMD overhead likely exceeded benefits for typical Solidity token lengths
/// - Added complexity without measurable gains
/// - Required nightly Rust and `portable_simd` feature
/// 
/// ## Lessons learned:
/// - SIMD optimizations need large data chunks to amortize setup overhead
/// - Solidity identifiers/tokens are typically short (2-20 chars)
/// - Branch prediction on modern CPUs is very good for simple loops
/// - Compiler auto-vectorization may already be optimal for this use case

/// Alternative optimization ideas to explore:
/// 
/// 1. **Lookup Table Optimization**:
///    - Pre-compute character classification bitmaps
///    - Use 256-byte lookup tables for O(1) character checks
///    - Already implemented in `char_class_table.rs`
/// 
/// 2. **Chunked Processing Without SIMD**:
///    - Process 4-8 bytes at a time using u32/u64 operations
///    - Use bit manipulation tricks for parallel byte comparisons
///    - Less overhead than SIMD, better than byte-by-byte
/// 
/// 3. **Specialized Fast Paths**:
///    - Detect common patterns (all spaces, common keywords)
///    - Use memchr for finding specific delimiters
///    - Already partially implemented
/// 
/// 4. **Memory Layout Optimization**:
///    - Better cache locality for token streams
///    - Arena allocation patterns
///    - Prefetch hints for large files
/// 
/// 5. **Algorithm-Level Improvements**:
///    - Reduce redundant character classification calls
///    - Optimize cursor advancement patterns
///    - Better integration with parser lookahead

pub mod attempted_optimizations {
    //! Documentation of specific optimization attempts and their results
    
    /// SIMD attempt using portable_simd - performance regression
    pub const SIMD_PORTABLE_ATTEMPT: &str = "
    Attempted 2025-01-06: True SIMD using portable_simd
    - Vectorized character classification
    - 16-byte parallel processing
    - Bitmask operations
    Result: Performance regression, reverted
    ";
    
    /// Chunked processing approach - reverted due to regressions
    pub const CHUNKED_PROCESSING_ATTEMPT: &str = "
    Attempted 2025-01-06: Chunked u32/u64 processing without SIMD overhead
    - Process 4-8 bytes at a time
    - Fast path for common patterns (spaces)
    - Hybrid memchr integration for long whitespace runs  
    - Ultra-aggressive SWAR bit manipulation with u64 operations
    Result: Performance regressions in tests, reverted
    ";
    
    /// SWAR bit manipulation attempt - reverted due to correctness issues
    pub const SWAR_ATTEMPT: &str = "
    Attempted 2025-01-06: SWAR (SIMD Within A Register) bit manipulation
    - u64 word-level processing with unsafe unaligned reads
    - Parallel bit arithmetic for character validation
    - Bit-parallel comparison techniques from base64 decoder blog
    - Pattern matching for common whitespace sequences
    Result: Significant correctness issues, test failures, reverted
    ";
    
    /// Future optimization candidates
    pub const NEXT_CANDIDATES: &[&str] = &[
        "Hand-tuned assembly for hot paths",
        "Better memchr integration for delimiter detection", 
        "Optimized cursor advancement patterns",
        "Cache-friendly token stream layouts",
        "Profile-guided optimization approaches",
    ];
}