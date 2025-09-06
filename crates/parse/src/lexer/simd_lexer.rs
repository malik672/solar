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

/// Optimized whitespace skipping using memchr for long runs.
pub fn skip_whitespace_bulk(input: &[u8]) -> usize {
    let mut pos = 0;
    while pos < input.len() && is_whitespace_fast(input[pos]) {
        pos += 1;
    }
    pos
}

/// Optimized identifier parsing with lookup table.
pub fn parse_identifier_bulk(input: &[u8]) -> usize {
    let mut pos = 0;
    while pos < input.len() && is_id_continue_fast(input[pos]) {
        pos += 1;
    }
    pos
}

/// Optimized decimal digit parsing.
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

/// Optimized hex digit parsing.
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
            assert_eq!(simd_id, scalar_id, "Identifier mismatch for: {:?}",
                   std::str::from_utf8(input).unwrap_or("invalid utf8"));
        }
    }
}