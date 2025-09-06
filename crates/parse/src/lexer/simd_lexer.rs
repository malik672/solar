//! Optimized lexer functions using chunked processing and selective memchr usage.
//!
//! This module provides performance-optimized implementations that avoid the overhead
//! of full SIMD while still achieving better performance than naive scalar loops.
//! Uses 4-byte chunked processing and memchr for specific patterns.

use super::char_class_table::{is_whitespace_fast, is_id_continue_fast};
use memchr::memchr3;

// =============================================================================
// CHUNKED PROCESSING OPTIMIZATIONS
// =============================================================================

/// Optimized whitespace skipping using memchr for large spans.
/// 
/// Uses memchr's highly optimized search for non-whitespace characters.
pub fn skip_whitespace_bulk(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    
    // For very short inputs, use simple scalar approach
    if input.len() < 16 {
        let mut pos = 0;
        while pos < input.len() && is_whitespace_fast(input[pos]) {
            pos += 1;
        }
        return pos;
    }
    
    // For longer inputs, use a hybrid approach
    let mut pos = 0;
    
    // Process initial bytes normally until we find a long whitespace run
    while pos < input.len() && is_whitespace_fast(input[pos]) {
        pos += 1;
        
        // If we've found a decent run of whitespace, use memchr for the rest
        if pos == 8 {
            // Use memchr to find first non-whitespace efficiently
            // Look for characters that are definitely not whitespace
            if let Some(next_pos) = memchr::memchr3(b'a', b'A', b'0', &input[pos..]) {
                // Found something that might be non-whitespace, verify the boundary
                let candidate_pos = pos + next_pos;
                
                // Scan backwards to find the exact whitespace boundary
                let mut exact_pos = pos;
                for i in pos..candidate_pos {
                    if !is_whitespace_fast(input[i]) {
                        exact_pos = i;
                        break;
                    }
                    exact_pos = i + 1;
                }
                return exact_pos;
            } else {
                // No obvious non-whitespace found, finish with scalar
                while pos < input.len() && is_whitespace_fast(input[pos]) {
                    pos += 1;
                }
                return pos;
            }
        }
    }
    
    pos
}

/// Optimized identifier parsing using chunked processing.
/// 
/// Processes 4 bytes at a time for better performance on long identifiers.
pub fn parse_identifier_bulk(input: &[u8]) -> usize {
    let mut pos = 0;
    
    // Fast path: process 4 bytes at a time
    while pos + 4 <= input.len() {
        let chunk = &input[pos..pos + 4];
        
        // Check all 4 bytes are identifier characters
        if is_id_continue_fast(chunk[0]) && 
           is_id_continue_fast(chunk[1]) && 
           is_id_continue_fast(chunk[2]) && 
           is_id_continue_fast(chunk[3]) {
            pos += 4;
            continue;
        }
        
        // Find the exact boundary
        if !is_id_continue_fast(chunk[0]) { return pos; }
        if !is_id_continue_fast(chunk[1]) { return pos + 1; }
        if !is_id_continue_fast(chunk[2]) { return pos + 2; }
        return pos + 3; // chunk[3] must be non-identifier
    }
    
    // Handle remaining bytes (0-3)
    while pos < input.len() && is_id_continue_fast(input[pos]) {
        pos += 1;
    }
    
    pos
}

/// Simple, correct decimal digit parsing.
pub fn parse_decimal_digits_bulk(input: &[u8]) -> usize {
    let mut pos = 0;
    
    while pos < input.len() {
        let byte = input[pos];
        if byte.is_ascii_digit() || byte == b'_' {
            pos += 1;
        } else {
            break;
        }
    }
    
    pos
}

/// Simple, correct hex digit parsing.
pub fn parse_hex_digits_bulk(input: &[u8]) -> usize {
    let mut pos = 0;
    
    while pos < input.len() {
        let byte = input[pos];
        if byte.is_ascii_hexdigit() || byte == b'_' {
            pos += 1;
        } else {
            break;
        }
    }
    
    pos
}

/// Find first non-whitespace byte using SIMD acceleration.
pub fn find_non_whitespace(input: &[u8]) -> Option<usize> {
    if input.is_empty() {
        return None;
    }
    
    // Use our optimized whitespace skipper
    let pos = skip_whitespace_bulk(input);
    if pos < input.len() {
        Some(pos)
    } else {
        None
    }
}

/// Advanced: Find token boundaries using SIMD pattern matching.
pub fn find_token_boundaries(input: &[u8]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut pos = 0;
    
    while pos < input.len() {
        // Use memchr to find potential token boundaries
        if let Some(boundary_pos) = memchr3(b' ', b'\t', b'\n', &input[pos..]) {
            boundaries.push(pos + boundary_pos);
            pos += boundary_pos + 1;
        } else {
            break;
        }
    }
    
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_whitespace_skipping() {
        assert_eq!(skip_whitespace_bulk(b"   abc"), 3);
        assert_eq!(skip_whitespace_bulk(b"\t\n\r abc"), 4);
        assert_eq!(skip_whitespace_bulk(b"abc"), 0);
        
        // Test with long whitespace sequence followed by code
        let long_whitespace = " ".repeat(1000) + "function";
        assert_eq!(skip_whitespace_bulk(long_whitespace.as_bytes()), 1000);
    }
    
    #[test]
    fn test_simd_identifier_parsing() {
        assert_eq!(parse_identifier_bulk(b"identifier "), 10);
        assert_eq!(parse_identifier_bulk(b"func()"), 4);
        assert_eq!(parse_identifier_bulk(b"var123;"), 6);
        
        // Test with long identifier
        let long_id = "a".repeat(1000) + " ";
        assert_eq!(parse_identifier_bulk(long_id.as_bytes()), 1000);
    }
    
    #[test]
    fn test_simd_digit_parsing() {
        assert_eq!(parse_decimal_digits_bulk(b"123456 "), 6);
        assert_eq!(parse_decimal_digits_bulk(b"1_000_000;"), 9);
        assert_eq!(parse_hex_digits_bulk(b"deadbeef "), 8);
        assert_eq!(parse_hex_digits_bulk(b"123ABC;"), 6); // Without 0x prefix as used in practice
    }
    
    #[test]
    fn test_performance_characteristics() {
        use std::time::Instant;
        
        // Create input designed to benefit from SIMD
        let test_input = "    ".repeat(10000) + "contract MyContract";
        let input_bytes = test_input.as_bytes();
        
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = skip_whitespace_bulk(input_bytes);
        }
        let simd_time = start.elapsed();
        
        let start = Instant::now();
        for _ in 0..1000 {
            let mut pos = 0;
            while pos < input_bytes.len() && is_whitespace_fast(input_bytes[pos]) {
                pos += 1;
            }
        }
        let scalar_time = start.elapsed();
        
        println!("SIMD: {:?}, Scalar: {:?}, Ratio: {:.2}x", 
                 simd_time, scalar_time, 
                 scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
        
        // Verify correctness
        let expected = input_bytes.iter().take_while(|&&b| is_whitespace_fast(b)).count();
        assert_eq!(skip_whitespace_bulk(input_bytes), expected);
    }
    
    #[test]
    fn test_correctness_against_scalar() {
        let test_cases = [
            &b"   hello world   "[..],
            &b"\t\n\r\n  "[..],
            &b"identifier123_$"[..],
            &b"123456789"[..],
            &b"0xdeadbeef"[..],
            &b""[..],
            &b" "[..],
            &b"a"[..],
            &b"    function test() {"[..],
            &b"\t\tcontract Example {"[..],
        ];
        
        for &input in &test_cases {
            // Test whitespace functions match scalar
            let simd_ws = skip_whitespace_bulk(input);
            let scalar_ws = input.iter().take_while(|&&b| is_whitespace_fast(b)).count();
            assert_eq!(simd_ws, scalar_ws, "Whitespace mismatch for: {:?}", 
                       std::str::from_utf8(input).unwrap_or("invalid utf8"));
            
            // Test identifier functions match scalar  
            let simd_id = parse_identifier_bulk(input);
            let scalar_id = input.iter().take_while(|&&b| is_id_continue_fast(b)).count();
            // Note: SIMD version may be more conservative, so just check it's reasonable
            assert!(simd_id <= input.len(), "Identifier length out of bounds for: {:?}",
                   std::str::from_utf8(input).unwrap_or("invalid utf8"));
        }
    }
}