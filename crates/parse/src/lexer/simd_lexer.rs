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

/// Ultra-optimized whitespace skipping using unrolled loops and branch elimination.
/// 
/// Uses aggressive unrolling and pattern matching for maximum throughput.
pub fn skip_whitespace_bulk(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    
    let mut pos = 0;
    let len = input.len();
    
    // Ultra-aggressive unrolled loop for maximum performance
    while pos + 16 <= len {
        let chunk = &input[pos..pos + 16];
        
        // Check common patterns first (branch predictor friendly)
        if chunk == b"                " { // 16 spaces
            pos += 16;
            continue;
        }
        
        if chunk == b"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t" { // 16 tabs  
            pos += 16;
            continue;
        }
        
        // Unrolled loop for maximum performance - no branches in inner loop
        let mut i = 0;
        while i < 16 {
            if !is_whitespace_fast(chunk[i]) {
                return pos + i;
            }
            i += 1;
        }
        pos += 16;
    }
    
    // Handle remaining bytes with 8-byte unrolling
    while pos + 8 <= len {
        let chunk = &input[pos..pos + 8];
        
        if chunk == b"        " { // 8 spaces
            pos += 8;
            continue;
        }
        
        let mut i = 0;
        while i < 8 {
            if !is_whitespace_fast(chunk[i]) {
                return pos + i;
            }
            i += 1;
        }
        pos += 8;
    }
    
    // Handle remaining bytes with 4-byte unrolling
    while pos + 4 <= len {
        let chunk = &input[pos..pos + 4];
        
        if chunk == b"    " { // 4 spaces
            pos += 4;
            continue;
        }
        
        // Unrolled 4-byte check
        if !is_whitespace_fast(chunk[0]) { return pos; }
        if !is_whitespace_fast(chunk[1]) { return pos + 1; }
        if !is_whitespace_fast(chunk[2]) { return pos + 2; }
        if !is_whitespace_fast(chunk[3]) { return pos + 3; }
        pos += 4;
    }
    
    // Handle final bytes
    while pos < len && is_whitespace_fast(input[pos]) {
        pos += 1;
    }
    
    pos
}

/// Ultra-optimized identifier parsing using unrolled loops and lookup tables.
/// 
/// Uses aggressive unrolling for maximum identifier parsing speed.
pub fn parse_identifier_bulk(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    
    let mut pos = 0;
    let len = input.len();
    
    // Ultra-aggressive unrolled loop for identifiers - most common tokens
    while pos + 16 <= len {
        let chunk = &input[pos..pos + 16];
        
        // Unrolled 16-byte check - no branches in tight loop
        if !is_id_continue_fast(chunk[0]) { return pos; }
        if !is_id_continue_fast(chunk[1]) { return pos + 1; }
        if !is_id_continue_fast(chunk[2]) { return pos + 2; }
        if !is_id_continue_fast(chunk[3]) { return pos + 3; }
        if !is_id_continue_fast(chunk[4]) { return pos + 4; }
        if !is_id_continue_fast(chunk[5]) { return pos + 5; }
        if !is_id_continue_fast(chunk[6]) { return pos + 6; }
        if !is_id_continue_fast(chunk[7]) { return pos + 7; }
        if !is_id_continue_fast(chunk[8]) { return pos + 8; }
        if !is_id_continue_fast(chunk[9]) { return pos + 9; }
        if !is_id_continue_fast(chunk[10]) { return pos + 10; }
        if !is_id_continue_fast(chunk[11]) { return pos + 11; }
        if !is_id_continue_fast(chunk[12]) { return pos + 12; }
        if !is_id_continue_fast(chunk[13]) { return pos + 13; }
        if !is_id_continue_fast(chunk[14]) { return pos + 14; }
        if !is_id_continue_fast(chunk[15]) { return pos + 15; }
        pos += 16;
    }
    
    // 8-byte unrolled processing
    while pos + 8 <= len {
        let chunk = &input[pos..pos + 8];
        
        // Unrolled 8-byte check
        if !is_id_continue_fast(chunk[0]) { return pos; }
        if !is_id_continue_fast(chunk[1]) { return pos + 1; }
        if !is_id_continue_fast(chunk[2]) { return pos + 2; }
        if !is_id_continue_fast(chunk[3]) { return pos + 3; }
        if !is_id_continue_fast(chunk[4]) { return pos + 4; }
        if !is_id_continue_fast(chunk[5]) { return pos + 5; }
        if !is_id_continue_fast(chunk[6]) { return pos + 6; }
        if !is_id_continue_fast(chunk[7]) { return pos + 7; }
        pos += 8;
    }
    
    // 4-byte unrolled processing 
    while pos + 4 <= len {
        let chunk = &input[pos..pos + 4];
        
        // Unrolled 4-byte check
        if !is_id_continue_fast(chunk[0]) { return pos; }
        if !is_id_continue_fast(chunk[1]) { return pos + 1; }
        if !is_id_continue_fast(chunk[2]) { return pos + 2; }
        if !is_id_continue_fast(chunk[3]) { return pos + 3; }
        pos += 4;
    }
    
    // Handle remaining bytes
    while pos < len && is_id_continue_fast(input[pos]) {
        pos += 1;
    }
    
    pos
}

/// Ultra-optimized decimal digit parsing with unrolled loops.
/// 
/// Uses aggressive unrolling for maximum digit parsing speed.
pub fn parse_decimal_digits_bulk(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    
    let mut pos = 0;
    let len = input.len();
    
    // Ultra-aggressive unrolled digit parsing
    while pos + 8 <= len {
        let chunk = &input[pos..pos + 8];
        
        // Unrolled 8-byte digit check
        if !is_digit_or_underscore(chunk[0]) { return pos; }
        if !is_digit_or_underscore(chunk[1]) { return pos + 1; }
        if !is_digit_or_underscore(chunk[2]) { return pos + 2; }
        if !is_digit_or_underscore(chunk[3]) { return pos + 3; }
        if !is_digit_or_underscore(chunk[4]) { return pos + 4; }
        if !is_digit_or_underscore(chunk[5]) { return pos + 5; }
        if !is_digit_or_underscore(chunk[6]) { return pos + 6; }
        if !is_digit_or_underscore(chunk[7]) { return pos + 7; }
        pos += 8;
    }
    
    // Handle remaining bytes
    while pos < len {
        let byte = input[pos];
        if is_digit_or_underscore(byte) {
            pos += 1;
        } else {
            break;
        }
    }
    
    pos
}

#[inline(always)]
fn is_digit_or_underscore(byte: u8) -> bool {
    byte.is_ascii_digit() || byte == b'_'
}

/// Ultra-optimized hex digit parsing with unrolled loops.
/// 
/// Uses aggressive unrolling for maximum hex digit parsing speed.
pub fn parse_hex_digits_bulk(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    
    let mut pos = 0;
    let len = input.len();
    
    // Ultra-aggressive unrolled hex digit parsing
    while pos + 8 <= len {
        let chunk = &input[pos..pos + 8];
        
        // Unrolled 8-byte hex check
        if !is_hex_or_underscore(chunk[0]) { return pos; }
        if !is_hex_or_underscore(chunk[1]) { return pos + 1; }
        if !is_hex_or_underscore(chunk[2]) { return pos + 2; }
        if !is_hex_or_underscore(chunk[3]) { return pos + 3; }
        if !is_hex_or_underscore(chunk[4]) { return pos + 4; }
        if !is_hex_or_underscore(chunk[5]) { return pos + 5; }
        if !is_hex_or_underscore(chunk[6]) { return pos + 6; }
        if !is_hex_or_underscore(chunk[7]) { return pos + 7; }
        pos += 8;
    }
    
    // Handle remaining bytes
    while pos < len {
        let byte = input[pos];
        if is_hex_or_underscore(byte) {
            pos += 1;
        } else {
            break;
        }
    }
    
    pos
}

#[inline(always)]
fn is_hex_or_underscore(byte: u8) -> bool {
    byte.is_ascii_hexdigit() || byte == b'_'
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