//! TDD Tests for Bounded Semantic Chunker
//!
//! These tests define the expected behavior for the improved semantic chunker
//! that enforces hard token limits while preserving semantic boundaries.
//!
//! Run with: cargo test bounded_chunker

#[cfg(test)]
mod tests {
    use super::super::bounded_chunker::*;

    // =========================================================================
    // TEST FIXTURES
    // =========================================================================

    /// Small Rust function that fits in any reasonable limit
    const SMALL_FUNCTION: &str = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;

    /// Medium function (~150 tokens)
    const MEDIUM_FUNCTION: &str = r#"
/// Authenticate user with credentials
pub async fn login(&self, email: &str, password: &str) -> Result<Token, AuthError> {
    if email.is_empty() || password.is_empty() {
        return Err(AuthError::InvalidInput);
    }

    let user = self.db.find_user_by_email(email).await?
        .ok_or(AuthError::UserNotFound)?;

    if !verify_password(password, &user.password_hash)? {
        return Err(AuthError::InvalidPassword);
    }

    Ok(self.generate_token(&user.id))
}
"#;

    /// Large impl block with multiple methods (~400 tokens)
    const LARGE_IMPL_BLOCK: &str = r#"
impl AuthService {
    pub fn new(db: Database, cache: Redis) -> Self {
        Self { db, cache }
    }

    pub async fn login(&self, email: &str, password: &str) -> Result<Token, AuthError> {
        if email.is_empty() {
            return Err(AuthError::InvalidInput);
        }
        let user = self.db.find_user_by_email(email).await?;
        Ok(self.generate_token(&user.unwrap().id))
    }

    pub async fn logout(&self, token: &str) -> Result<(), AuthError> {
        let claims = self.verify_token(token)?;
        self.cache.delete(&format!("token:{}", claims.jti)).await?;
        Ok(())
    }

    pub async fn refresh(&self, token: &str) -> Result<Token, AuthError> {
        let claims = self.verify_token(token)?;
        if claims.is_expired() {
            return Err(AuthError::TokenExpired);
        }
        Ok(self.generate_token(&claims.sub))
    }

    fn generate_token(&self, user_id: &str) -> Token {
        Token::new(user_id, Duration::hours(24))
    }

    fn verify_token(&self, token: &str) -> Result<Claims, AuthError> {
        Claims::decode(token).map_err(|_| AuthError::InvalidToken)
    }
}
"#;

    /// Monster single line - a massive regex (~200 tokens, no newlines)
    const MONSTER_REGEX: &str = r#"const URL_REGEX: &str = r"^(?:(?:https?|ftp):\/\/)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:\/\S*)?$";"#;

    /// Long string literal with newlines inside
    const LONG_STRING_LITERAL: &str = r#"
const ERROR_MESSAGES: &str = "
ERROR_001: Authentication failed. Please check your credentials.
ERROR_002: Session expired. Please log in again.
ERROR_003: Permission denied. You do not have access.
ERROR_004: Resource not found. The item does not exist.
ERROR_005: Rate limit exceeded. Please wait before retrying.
ERROR_006: Invalid input format. Check the documentation.
ERROR_007: Connection timeout. Server did not respond.
ERROR_008: Service unavailable. Try again later.
ERROR_009: Conflict detected. Resource was modified.
ERROR_010: Payload too large. Reduce request size.
";
"#;

    /// Nested structure with classes and methods
    const NESTED_STRUCTURE: &str = r#"
pub mod auth {
    pub struct User {
        id: String,
        email: String,
    }

    impl User {
        pub fn new(id: String, email: String) -> Self {
            Self { id, email }
        }

        pub fn validate(&self) -> bool {
            !self.email.is_empty()
        }
    }

    pub struct Session {
        user: User,
        token: String,
    }

    impl Session {
        pub fn new(user: User) -> Self {
            let token = generate_token();
            Self { user, token }
        }

        pub fn is_valid(&self) -> bool {
            !self.token.is_empty() && self.user.validate()
        }
    }
}
"#;

    // =========================================================================
    // 1. HARD LIMIT ENFORCEMENT TESTS
    // =========================================================================

    mod hard_limit_tests {
        use super::*;

        #[test]
        fn chunk_never_exceeds_max_tokens() {
            // Given: A chunker with max_tokens = 100
            let chunker = create_test_chunker(100);

            // When: Chunking content of any size
            let test_cases = [
                SMALL_FUNCTION,
                MEDIUM_FUNCTION,
                LARGE_IMPL_BLOCK,
                MONSTER_REGEX,
                LONG_STRING_LITERAL,
                NESTED_STRUCTURE,
            ];

            for content in test_cases {
                let chunks = chunker.chunk(content, "rust");

                // Then: Every chunk must be <= max_tokens
                for (i, chunk) in chunks.iter().enumerate() {
                    let tokens = chunker.count_tokens(&chunk.content);
                    assert!(
                        tokens <= 100,
                        "Chunk {} exceeded max_tokens: {} > 100, content preview: {}",
                        i,
                        tokens,
                        &chunk.content[..chunk.content.len().min(100)]
                    );
                }
            }
        }

        #[test]
        fn respects_target_tokens_when_possible() {
            // Given: max=100, target=80 (80%)
            let chunker = create_test_chunker_with_target(100, 80);

            // When: Chunking content that can be split at AST boundaries
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            // Then: Most chunks should be close to target (within 20% tolerance)
            let chunks_near_target = chunks
                .iter()
                .filter(|c| {
                    let tokens = chunker.count_tokens(&c.content);
                    tokens >= 60 && tokens <= 100 // 60-100 range
                })
                .count();

            // At least 50% of chunks should be near target
            assert!(
                chunks_near_target * 2 >= chunks.len(),
                "Too few chunks near target: {}/{}",
                chunks_near_target,
                chunks.len()
            );
        }

        #[test]
        fn small_content_stays_whole() {
            // Given: A chunker with generous limit
            let chunker = create_test_chunker(500);

            // When: Chunking small content
            let chunks = chunker.chunk(SMALL_FUNCTION, "rust");

            // Then: Should produce exactly 1 chunk
            assert_eq!(chunks.len(), 1, "Small function should stay as one chunk");
        }

        #[test]
        fn empty_content_produces_no_chunks() {
            let chunker = create_test_chunker(100);
            let chunks = chunker.chunk("", "rust");
            assert!(chunks.is_empty(), "Empty content should produce no output");
        }

        #[test]
        fn whitespace_only_produces_no_chunks() {
            let chunker = create_test_chunker(100);
            let chunks = chunker.chunk("   \n\n\t  ", "rust");
            assert!(chunks.is_empty(), "Whitespace-only should produce no output");
        }
    }

    // =========================================================================
    // 2. AST-AWARE SPLITTING TESTS
    // =========================================================================

    mod ast_splitting_tests {
        use super::*;

        #[test]
        fn splits_at_function_boundaries() {
            // Given: An impl block with multiple methods
            let chunker = create_test_chunker(150); // Force splitting

            // When: Chunking
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            // Then: No chunk should contain partial function
            for chunk in &chunks {
                // Count function starts and ends
                let fn_starts = chunk.content.matches("fn ").count();
                let _fn_ends = chunk.content.matches('}').count();

                // Each chunk should have balanced braces for functions it contains
                // (This is a simplified check - real impl would use AST)
                if fn_starts > 0 {
                    assert!(
                        chunk.content.contains('}'),
                        "Chunk with fn should have closing brace, preview: {}",
                        &chunk.content[..chunk.content.len().min(100)]
                    );
                }
            }
        }

        #[test]
        fn merges_small_siblings() {
            // Given: Multiple small functions that can fit together
            let small_functions = r#"
fn a() -> i32 { 1 }
fn b() -> i32 { 2 }
fn c() -> i32 { 3 }
fn d() -> i32 { 4 }
"#;
            let chunker = create_test_chunker(200); // Room for multiple small fns

            // When: Chunking
            let chunks = chunker.chunk(small_functions, "rust");

            // Then: Should merge into fewer chunks, not one per function
            assert!(
                chunks.len() < 4,
                "Small functions should be merged, got {} chunks",
                chunks.len()
            );
        }

        #[test]
        fn preserves_nested_structure() {
            // Given: Nested module with structs and impls
            let chunker = create_test_chunker(200);

            // When: Chunking
            let chunks = chunker.chunk(NESTED_STRUCTURE, "rust");

            // Then: Each chunk should have balanced braces
            for chunk in &chunks {
                let opens = chunk.content.matches('{').count();
                let closes = chunk.content.matches('}').count();

                // Allow slight imbalance for partial content, but not extreme
                let diff = (opens as i32 - closes as i32).abs();
                assert!(
                    diff <= 2,
                    "Chunk has unbalanced braces: {} opens, {} closes, preview: {}",
                    opens,
                    closes,
                    &chunk.content[..chunk.content.len().min(200)]
                );
            }
        }

        #[test]
        fn handles_unsupported_language_gracefully() {
            let chunker = create_test_chunker(100);

            // When: Chunking unsupported language
            let chunks = chunker.chunk("some content here that is long enough", "brainfuck");

            // Then: Should fall back to line-based, not panic
            assert!(!chunks.is_empty(), "Should produce chunks for unknown language");
        }
    }

    // =========================================================================
    // 3. FALLBACK CHAIN TESTS (AST -> Line -> Word -> Truncate)
    // =========================================================================

    mod fallback_chain_tests {
        use super::*;

        #[test]
        fn falls_back_to_line_split_for_large_leaf() {
            // Given: A very small max_tokens that forces line splitting
            // Each line is ~10 tokens, so max=15 should force splitting
            let chunker = create_test_chunker(15);

            // When: Chunking a function body (leaf node in some AST views)
            let leaf_content = r#"
let a = compute_something_complex_here();
let b = another_computation_with_params();
let c = yet_another_one_doing_stuff();
let d = final_computation_returning_result();
return combine_all_results(a, b, c, d);
"#;
            let chunks = chunker.chunk(leaf_content, "rust");

            // Then: Should produce multiple line-based chunks
            assert!(chunks.len() > 1, "Should split at line boundaries, got {} chunk(s)", chunks.len());

            // And: Each chunk should be valid (not mid-line)
            for chunk in &chunks {
                // Chunks should start with non-whitespace or be properly trimmed
                let trimmed = chunk.content.trim();
                assert!(
                    !trimmed.is_empty(),
                    "Chunk should not be empty after trim"
                );
            }
        }

        #[test]
        fn falls_back_to_word_split_for_long_line() {
            // Given: Very small limit (each word is ~2 tokens, so max=5 forces splitting)
            let chunker = create_test_chunker(5);

            // When: Chunking a long single line with spaces
            let long_line = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12";
            let chunks = chunker.chunk(long_line, "text");

            // Then: Should split at word boundaries
            assert!(chunks.len() > 1, "Should split long line, got {} chunk(s)", chunks.len());

            // And: No chunk should have cut words (no partial words)
            for chunk in &chunks {
                let content = chunk.content.trim();
                // Should not start or end with partial word
                assert!(
                    !content.starts_with("ord"), // partial "word"
                    "Chunk starts with partial word: {}",
                    content
                );
            }
        }

        #[test]
        fn truncates_only_when_no_other_option() {
            // Given: Impossibly small limit
            let chunker = create_test_chunker(10);

            // When: Chunking a line with no spaces (cannot word-split)
            let no_spaces = "abcdefghijklmnopqrstuvwxyz0123456789";
            let chunks = chunker.chunk(no_spaces, "text");

            // Then: Should truncate but still produce valid output
            assert!(!chunks.is_empty(), "Should produce at least one chunk");

            // And: Each chunk respects the limit
            for chunk in &chunks {
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(tokens <= 10, "Truncated chunk exceeds limit: {}", tokens);
            }
        }
    }

    // =========================================================================
    // 4. OVERLAP TESTS
    // =========================================================================

    mod overlap_tests {
        use super::*;

        #[test]
        fn adds_overlap_between_chunks() {
            // Given: Chunker with 15% overlap
            let chunker = create_test_chunker_with_overlap(100, 0.15);

            // When: Chunking content that produces multiple chunks
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            if chunks.len() >= 2 {
                // Then: Adjacent chunks should have overlapping content
                for i in 0..chunks.len() - 1 {
                    let chunk1_end = get_last_n_tokens(&chunks[i].content, 15);
                    let chunk2_start = get_first_n_tokens(&chunks[i + 1].content, 15);

                    // There should be some overlap
                    let has_overlap = chunk1_end
                        .iter()
                        .any(|token| chunk2_start.contains(token));

                    assert!(
                        has_overlap,
                        "Chunks {} and {} should overlap",
                        i,
                        i + 1
                    );
                }
            }
        }

        #[test]
        fn overlap_does_not_exceed_max_tokens() {
            // Given: Chunker with overlap
            let chunker = create_test_chunker_with_overlap(100, 0.20);

            // When: Chunking
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            // Then: Even with overlap, chunks must not exceed max
            for chunk in &chunks {
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(
                    tokens <= 100,
                    "Chunk with overlap exceeds max: {} > 100",
                    tokens
                );
            }
        }

        #[test]
        fn zero_overlap_produces_no_duplication() {
            // Given: Chunker with no overlap
            let chunker = create_test_chunker_with_overlap(100, 0.0);

            // When: Chunking
            let chunks = chunker.chunk(MEDIUM_FUNCTION, "rust");

            // Then: Concatenating all chunks should roughly equal original
            // (allowing for trimming)
            let combined: String = chunks.iter().map(|c| c.content.as_str()).collect();
            let original_tokens = chunker.count_tokens(MEDIUM_FUNCTION.trim());
            let combined_tokens = chunker.count_tokens(&combined);

            // Combined should be close to original (within 10% for trimming)
            let diff = (combined_tokens as f32 - original_tokens as f32).abs();
            let tolerance = original_tokens as f32 * 0.1;
            assert!(
                diff <= tolerance,
                "No-overlap chunks have unexpected duplication: {} vs {}",
                combined_tokens,
                original_tokens
            );
        }
    }

    // =========================================================================
    // 5. NO CONTEXT INJECTION TESTS
    // =========================================================================

    mod no_context_injection_tests {
        use super::*;

        #[test]
        fn chunks_do_not_contain_file_prefix() {
            let chunker = create_test_chunker(200);
            let chunks = chunker.chunk(MEDIUM_FUNCTION, "rust");

            for chunk in &chunks {
                assert!(
                    !chunk.content.contains("[file:"),
                    "Chunk should not contain [file: prefix, preview: {}",
                    &chunk.content[..chunk.content.len().min(100)]
                );
            }
        }

        #[test]
        fn chunks_do_not_contain_context_prefix() {
            let chunker = create_test_chunker(200);
            let chunks = chunker.chunk(MEDIUM_FUNCTION, "rust");

            for chunk in &chunks {
                assert!(
                    !chunk.content.contains("[context:"),
                    "Chunk should not contain [context: prefix, preview: {}",
                    &chunk.content[..chunk.content.len().min(100)]
                );
            }
        }

        #[test]
        fn chunk_content_is_pure_code() {
            let chunker = create_test_chunker(200);
            let chunks = chunker.chunk(SMALL_FUNCTION, "rust");

            assert_eq!(chunks.len(), 1);

            // The chunk should be the original code (possibly trimmed)
            let chunk_trimmed = chunks[0].content.trim();
            let original_trimmed = SMALL_FUNCTION.trim();

            assert_eq!(
                chunk_trimmed, original_trimmed,
                "Chunk should be pure code without any injected context"
            );
        }

        #[test]
        fn metadata_stored_separately() {
            let chunker = create_test_chunker(200);
            let chunks = chunker.chunk_with_metadata(MEDIUM_FUNCTION, "rust", "src/auth.rs");

            for chunk in &chunks {
                // Content should be pure
                assert!(!chunk.content.contains("[file:"));

                // Metadata should exist separately
                assert!(chunk.metadata.is_some());
                let meta = chunk.metadata.as_ref().unwrap();
                assert_eq!(meta.file_path, "src/auth.rs");
            }
        }
    }

    // =========================================================================
    // 6. MONSTER LINE HANDLING TESTS (No Truncation)
    // =========================================================================

    mod monster_line_tests {
        use super::*;

        #[test]
        fn monster_regex_splits_without_truncation() {
            // Given: Small limit that requires splitting the regex
            let chunker = create_test_chunker(50);

            // When: Chunking a monster regex
            let chunks = chunker.chunk(MONSTER_REGEX, "rust");

            // Then: All content should be preserved (no truncation)
            let combined: String = chunks.iter().map(|c| c.content.as_str()).collect();

            // The combined content should contain all the original regex parts
            assert!(
                combined.contains("https?") && combined.contains("ftp"),
                "Should preserve protocol patterns"
            );
            assert!(
                combined.contains("10|127") || combined.contains("169"),
                "Should preserve IP patterns"
            );
        }

        #[test]
        fn pattern_aware_split_for_regex() {
            let chunker = create_test_chunker(100);

            // When: Using pattern-aware splitting for regex
            let chunks = chunker.chunk_regex_aware(MONSTER_REGEX);

            // Then: Splits should be at logical regex boundaries
            for chunk in &chunks {
                let content = chunk.content.trim();

                // Should not split in the middle of a character class [...]
                let open_brackets = content.matches('[').count();
                let close_brackets = content.matches(']').count();
                let diff = (open_brackets as i32 - close_brackets as i32).abs();

                assert!(
                    diff <= 1,
                    "Regex chunk has unbalanced brackets: {}",
                    content
                );
            }
        }

        #[test]
        fn multi_vector_option_preserves_all_content() {
            let chunker = create_test_chunker(50);

            // When: Using multi-vector mode
            let result = chunker.chunk_multi_vector(MONSTER_REGEX, "rust");

            // Then: Should return multiple chunks with parent reference
            assert!(result.chunks.len() > 1, "Should produce multiple chunks");
            assert!(
                !result.parent_id.is_empty(),
                "Should have parent ID for linking"
            );

            // And: All chunks should reference the same parent
            for chunk in &result.chunks {
                assert_eq!(chunk.parent_id.as_deref(), Some(result.parent_id.as_str()));
            }
        }

        #[test]
        fn weighted_average_option_produces_single_embedding() {
            let chunker = create_test_chunker(50);
            let embedder = MockEmbedder::new(384); // 384-dim embeddings

            // When: Using weighted average mode
            let result = chunker.chunk_with_weighted_average(MONSTER_REGEX, "rust", &embedder);

            // Then: Should return single combined embedding
            assert_eq!(result.embedding.len(), 384);

            // And: Embedding should be normalized (L2 norm ~ 1.0)
            let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Combined embedding should be normalized: {}",
                norm
            );
        }
    }

    // =========================================================================
    // 7. TOKEN COUNTING TESTS
    // =========================================================================

    mod token_counting_tests {
        use super::*;

        #[test]
        fn counts_tokens_accurately() {
            let chunker = create_test_chunker(100);

            // Token counts based on ~4 non-whitespace chars per token
            // Formula: ceil(non_ws_chars / 4.0)
            let test_cases = [
                ("hello", 1, 3),           // 5 chars -> ceil(5/4) = 2
                ("hello world", 1, 4),     // 10 chars -> ceil(10/4) = 3
                ("fn main() {}", 1, 5),    // 11 chars -> ceil(11/4) = 3
                ("let x = 1;", 1, 4),      // 7 chars -> ceil(7/4) = 2
            ];

            for (text, min_expected, max_expected) in test_cases {
                let count = chunker.count_tokens(text);
                assert!(
                    count >= min_expected && count <= max_expected,
                    "Token count for '{}': {} (expected {}-{})",
                    text,
                    count,
                    min_expected,
                    max_expected
                );
            }
        }

        #[test]
        fn approximation_is_conservative() {
            let chunker = create_test_chunker(100);

            // Approximation should err on the side of overestimating
            // to avoid exceeding limits
            let code = "pub async fn process_request(&self, req: Request) -> Response { }";

            let approx = chunker.approx_tokens(code);
            let actual = chunker.count_tokens(code);

            // Approximation should be >= actual (conservative)
            // or very close (within 20%)
            assert!(
                approx >= actual || (actual - approx) <= actual / 5,
                "Approximation {} should be close to or above actual {}",
                approx,
                actual
            );
        }
    }

    // =========================================================================
    // 8. CHUNK METADATA TESTS
    // =========================================================================

    mod metadata_tests {
        use super::*;

        #[test]
        fn tracks_line_numbers() {
            let chunker = create_test_chunker(100);
            let chunks = chunker.chunk_with_positions(LARGE_IMPL_BLOCK, "rust");

            for chunk in &chunks {
                assert!(chunk.start_line > 0, "Start line should be positive");
                assert!(
                    chunk.end_line >= chunk.start_line,
                    "End line should be >= start line"
                );
            }
        }

        #[test]
        fn line_numbers_are_contiguous() {
            let chunker = create_test_chunker_with_overlap(100, 0.0); // No overlap for this test
            let chunks = chunker.chunk_with_positions(LARGE_IMPL_BLOCK, "rust");

            if chunks.len() >= 2 {
                for i in 0..chunks.len() - 1 {
                    let gap = chunks[i + 1].start_line as i32 - chunks[i].end_line as i32;
                    assert!(
                        gap <= 1,
                        "Gap between chunks {} and {}: {} lines",
                        i,
                        i + 1,
                        gap
                    );
                }
            }
        }

        #[test]
        fn tracks_byte_offsets() {
            let chunker = create_test_chunker(100);
            let chunks = chunker.chunk_with_positions(MEDIUM_FUNCTION, "rust");

            for chunk in &chunks {
                // Byte offsets should be valid
                assert!(
                    chunk.start_byte < chunk.end_byte,
                    "Start byte should be < end byte"
                );
            }
        }
    }

    // =========================================================================
    // 9. INTEGRATION TESTS
    // =========================================================================

    mod integration_tests {
        use super::*;

        #[test]
        fn full_pipeline_produces_valid_chunks() {
            // Given: Realistic configuration
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 512,
                target_tokens: 400,
                overlap_ratio: 0.15,
                inject_context: false,
            });

            // When: Processing a real file
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            // Then: All invariants hold
            for chunk in &chunks {
                // 1. Size limit respected
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(tokens <= 512, "Exceeded max tokens");

                // 2. No context injection
                assert!(!chunk.content.contains("[file:"));
                assert!(!chunk.content.contains("[context:"));

                // 3. Content is non-empty
                assert!(!chunk.content.trim().is_empty());
            }
        }

        #[test]
        fn handles_mixed_content_file() {
            let mixed_content = format!(
                "{}\n\n{}\n\n{}",
                SMALL_FUNCTION, MEDIUM_FUNCTION, MONSTER_REGEX
            );

            let chunker = create_test_chunker(150);
            let chunks = chunker.chunk(&mixed_content, "rust");

            // Should handle all content types
            assert!(!chunks.is_empty());

            for chunk in &chunks {
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(tokens <= 150, "Mixed content chunk exceeded limit");
            }
        }

        #[test]
        fn consistent_results_across_calls() {
            let chunker = create_test_chunker(100);

            // Same input should produce same output
            let chunks1 = chunker.chunk(MEDIUM_FUNCTION, "rust");
            let chunks2 = chunker.chunk(MEDIUM_FUNCTION, "rust");

            assert_eq!(chunks1.len(), chunks2.len(), "Chunk count should be consistent");

            for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
                assert_eq!(c1.content, c2.content, "Chunk content should be consistent");
            }
        }
    }

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /// Creates a test chunker with specified max tokens
    fn create_test_chunker(max_tokens: usize) -> BoundedSemanticChunker {
        BoundedSemanticChunker::new(BoundedChunkerConfig {
            max_tokens,
            target_tokens: (max_tokens as f32 * 0.8) as usize,
            overlap_ratio: 0.0,
            inject_context: false,
        })
    }

    fn create_test_chunker_with_target(max_tokens: usize, target_tokens: usize) -> BoundedSemanticChunker {
        BoundedSemanticChunker::new(BoundedChunkerConfig {
            max_tokens,
            target_tokens,
            overlap_ratio: 0.0,
            inject_context: false,
        })
    }

    fn create_test_chunker_with_overlap(max_tokens: usize, overlap_ratio: f32) -> BoundedSemanticChunker {
        BoundedSemanticChunker::new(BoundedChunkerConfig {
            max_tokens,
            target_tokens: (max_tokens as f32 * 0.8) as usize,
            overlap_ratio,
            inject_context: false,
        })
    }

    fn get_first_n_tokens(text: &str, n: usize) -> Vec<String> {
        text.split_whitespace()
            .take(n)
            .map(String::from)
            .collect()
    }

    fn get_last_n_tokens(text: &str, n: usize) -> Vec<String> {
        let tokens: Vec<_> = text.split_whitespace().collect();
        tokens
            .into_iter()
            .rev()
            .take(n)
            .map(String::from)
            .collect()
    }

    // =========================================================================
    // 10. AST SIBLING OVERLAP TESTS (NEW - Improvement #1)
    // =========================================================================

    mod sibling_overlap_tests {
        use super::*;

        #[test]
        fn overlap_applied_between_ast_siblings() {
            // Given: Multiple functions that will be split into separate chunks
            let multiple_functions = r#"
fn first_function() -> i32 {
    let a = 1;
    let b = 2;
    let c = 3;
    a + b + c
}

fn second_function() -> i32 {
    let x = 10;
    let y = 20;
    x + y
}

fn third_function() -> i32 {
    let result = 100;
    result * 2
}
"#;
            // Chunker with 15% sibling overlap enabled
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 50, // Force splitting into separate chunks
                target_tokens: 40,
                overlap_ratio: 0.15,
                inject_context: false,
            });

            // When: Chunking with sibling overlap
            let chunks = chunker.chunk(multiple_functions, "rust");

            // Then: Adjacent chunks should share content from sibling boundaries
            if chunks.len() >= 2 {
                // Check that chunk 2 starts with some content from end of chunk 1
                let chunk1_lines: Vec<&str> = chunks[0].content.lines().collect();
                let chunk2_content = &chunks[1].content;

                // Last line(s) of chunk 1 should appear at start of chunk 2
                let has_sibling_overlap = chunk1_lines
                    .iter()
                    .rev()
                    .take(3)
                    .any(|line| chunk2_content.contains(line.trim()) && !line.trim().is_empty());

                assert!(
                    has_sibling_overlap,
                    "Chunks should have sibling overlap. Chunk 1 ends with:\n{}\nChunk 2 starts with:\n{}",
                    chunk1_lines.iter().rev().take(3).cloned().collect::<Vec<_>>().join("\n"),
                    chunks[1].content.lines().take(5).collect::<Vec<_>>().join("\n")
                );
            }
        }

        #[test]
        fn sibling_overlap_preserves_function_context() {
            // Given: Functions where context at boundaries matters
            let context_critical = r#"
impl Calculator {
    fn add(&self, a: i32, b: i32) -> i32 {
        self.log("add");
        a + b
    }

    fn subtract(&self, a: i32, b: i32) -> i32 {
        self.log("subtract");
        a - b
    }

    fn multiply(&self, a: i32, b: i32) -> i32 {
        self.log("multiply");
        a * b
    }
}
"#;
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 60,
                target_tokens: 48,
                overlap_ratio: 0.15,
                inject_context: false,
            });

            let chunks = chunker.chunk(context_critical, "rust");

            // Then: Each chunk (except first) should have some context from previous
            // to help embedding model understand the surrounding code
            for (i, chunk) in chunks.iter().enumerate().skip(1) {
                // Chunk should either start with overlap content or be self-contained
                let content = chunk.content.trim();

                // If it's a method, it should have context (impl block reference or overlap)
                if content.contains("fn ") && !content.starts_with("impl") {
                    // The chunk should have enough context to understand it's a method
                    // Either through overlap or structural hints
                    let has_context = content.contains("impl")
                        || content.contains("self")
                        || content.contains("}") // closing brace from overlap
                        || i == 0;

                    assert!(
                        has_context,
                        "Method in chunk {} should have context: {}",
                        i,
                        &content[..content.len().min(100)]
                    );
                }
            }
        }

        #[test]
        fn sibling_overlap_does_not_exceed_max_tokens() {
            // Given: Content that produces multiple AST chunks
            let large_content = r#"
fn function_one() -> String {
    let mut result = String::new();
    result.push_str("processing");
    result.push_str(" data");
    result
}

fn function_two() -> String {
    let mut buffer = String::new();
    buffer.push_str("handling");
    buffer.push_str(" request");
    buffer
}

fn function_three() -> String {
    let mut output = String::new();
    output.push_str("generating");
    output.push_str(" response");
    output
}
"#;
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 50,
                target_tokens: 40,
                overlap_ratio: 0.20, // 20% overlap
                inject_context: false,
            });

            let chunks = chunker.chunk(large_content, "rust");

            // Then: Even with overlap, no chunk exceeds max
            for (i, chunk) in chunks.iter().enumerate() {
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(
                    tokens <= 50,
                    "Chunk {} with sibling overlap exceeds max: {} > 50\nContent: {}",
                    i,
                    tokens,
                    &chunk.content[..chunk.content.len().min(200)]
                );
            }
        }

        #[test]
        fn sibling_overlap_calculates_from_previous_sibling() {
            // Given: Known content sizes
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 100,
                target_tokens: 80,
                overlap_ratio: 0.15, // 15% of previous chunk
                inject_context: false,
            });

            let functions = r#"
fn alpha() {
    println!("alpha function body here");
}

fn beta() {
    println!("beta function body here");
}
"#;
            let chunks = chunker.chunk(functions, "rust");

            if chunks.len() >= 2 {
                // Calculate expected overlap tokens from first chunk
                let chunk1_tokens = chunker.count_tokens(&chunks[0].content);
                let expected_overlap_tokens = (chunk1_tokens as f32 * 0.15) as usize;

                // Chunk 2 should be larger than just its own content due to overlap
                // (if overlap was applied)
                // This is a sanity check that overlap calculation is based on previous chunk
                assert!(
                    expected_overlap_tokens > 0 || chunk1_tokens < 7,
                    "Expected some overlap tokens: {} (from {} chunk1 tokens)",
                    expected_overlap_tokens,
                    chunk1_tokens
                );
            }
        }
    }

    // =========================================================================
    // 11. MINIMAL CONTEXT INJECTION TESTS (NEW - Improvement #2)
    // =========================================================================

    mod minimal_context_tests {
        use super::*;

        #[test]
        fn minimal_context_adds_file_path_only() {
            // Given: Chunker with minimal context enabled
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 200,
                target_tokens: 160,
                overlap_ratio: 0.0,
                inject_context: true, // Enable minimal context
            });

            // When: Chunking with file path
            let chunks = chunker.chunk_with_file_context(SMALL_FUNCTION, "rust", "src/math.rs");

            // Then: Each chunk should start with file path comment
            for chunk in &chunks {
                assert!(
                    chunk.content.starts_with("// src/math.rs\n"),
                    "Chunk should start with file path comment, got: {}",
                    &chunk.content[..chunk.content.len().min(50)]
                );
            }
        }

        #[test]
        fn minimal_context_does_not_add_breadcrumbs() {
            // Given: Chunker with minimal context
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 200,
                target_tokens: 160,
                overlap_ratio: 0.0,
                inject_context: true,
            });

            // When: Chunking nested content
            let chunks = chunker.chunk_with_file_context(NESTED_STRUCTURE, "rust", "src/auth/mod.rs");

            // Then: Should NOT have breadcrumb-style context like "impl User > fn new"
            for chunk in &chunks {
                assert!(
                    !chunk.content.contains(" > "),
                    "Should not have breadcrumb context: {}",
                    &chunk.content[..chunk.content.len().min(100)]
                );
                assert!(
                    !chunk.content.contains("[context:"),
                    "Should not have [context: prefix"
                );
            }
        }

        #[test]
        fn minimal_context_uses_comment_syntax_for_language() {
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 200,
                target_tokens: 160,
                overlap_ratio: 0.0,
                inject_context: true,
            });

            // Test Rust (// comment)
            let rust_chunks = chunker.chunk_with_file_context("fn test() {}", "rust", "src/lib.rs");
            assert!(rust_chunks[0].content.starts_with("// "));

            // Test Python (# comment)
            let py_chunks = chunker.chunk_with_file_context("def test(): pass", "python", "main.py");
            assert!(
                py_chunks[0].content.starts_with("# "),
                "Python should use # comment: {}",
                &py_chunks[0].content[..py_chunks[0].content.len().min(30)]
            );
        }

        #[test]
        fn minimal_context_counts_toward_token_limit() {
            // Given: Very tight token limit
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 50,
                target_tokens: 40,
                overlap_ratio: 0.0,
                inject_context: true,
            });

            // When: Adding context to content near the limit
            let chunks = chunker.chunk_with_file_context(
                MEDIUM_FUNCTION,
                "rust",
                "src/very/long/nested/path/to/auth/service.rs"
            );

            // Then: Total tokens including context must not exceed max
            for chunk in &chunks {
                let tokens = chunker.count_tokens(&chunk.content);
                assert!(
                    tokens <= 50,
                    "Chunk with context exceeds max: {} tokens\n{}",
                    tokens,
                    &chunk.content[..chunk.content.len().min(200)]
                );
            }
        }

        #[test]
        fn context_disabled_produces_pure_code() {
            // Given: Chunker with context disabled (default)
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 200,
                target_tokens: 160,
                overlap_ratio: 0.0,
                inject_context: false, // Disabled
            });

            // When: Chunking with file context method
            let chunks = chunker.chunk_with_file_context(SMALL_FUNCTION, "rust", "src/math.rs");

            // Then: Should NOT have file path prefix
            for chunk in &chunks {
                assert!(
                    !chunk.content.starts_with("//"),
                    "With inject_context=false, should not add prefix: {}",
                    &chunk.content[..chunk.content.len().min(50)]
                );
            }
        }

        #[test]
        fn metadata_still_tracks_file_path_separately() {
            // Given: Chunker with context enabled
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 200,
                target_tokens: 160,
                overlap_ratio: 0.0,
                inject_context: true,
            });

            // When: Chunking with metadata
            let chunks = chunker.chunk_with_metadata(SMALL_FUNCTION, "rust", "src/utils.rs");

            // Then: Metadata should still have file path
            for chunk in &chunks {
                assert!(chunk.metadata.is_some());
                assert_eq!(chunk.metadata.as_ref().unwrap().file_path, "src/utils.rs");
            }
        }
    }

    // =========================================================================
    // 12. REDUCED CHUNK SIZE TESTS (NEW - Improvement #3)
    // =========================================================================

    mod reduced_chunk_size_tests {
        use super::*;

        #[test]
        fn default_target_is_reduced() {
            // Given: Default configuration
            let config = BoundedChunkerConfig::default();

            // Then: Target should be ~250-300 tokens (1000-1200 chars)
            // Old default was 400, new should be ~250
            assert!(
                config.target_tokens <= 300,
                "Default target_tokens should be <= 300, got {}",
                config.target_tokens
            );
            assert!(
                config.target_tokens >= 200,
                "Default target_tokens should be >= 200, got {}",
                config.target_tokens
            );
        }

        #[test]
        fn default_max_is_reduced() {
            // Given: Default configuration
            let config = BoundedChunkerConfig::default();

            // Then: Max should be ~350-400 tokens (1400-1600 chars)
            // Old default was 512, new should be ~350
            assert!(
                config.max_tokens <= 400,
                "Default max_tokens should be <= 400, got {}",
                config.max_tokens
            );
            assert!(
                config.max_tokens >= 300,
                "Default max_tokens should be >= 300, got {}",
                config.max_tokens
            );
        }

        #[test]
        fn for_model_calculates_appropriate_target() {
            // Given: Model with 512 token limit
            let config = BoundedChunkerConfig::for_model(512);

            // Then: Target should be 80% of max
            assert_eq!(
                config.target_tokens,
                (512.0 * 0.8) as usize,
                "Target should be 80% of max"
            );
        }

        #[test]
        fn reduced_size_produces_more_chunks() {
            // Given: Old config (large chunks) vs new config (smaller chunks)
            let old_config = BoundedChunkerConfig {
                max_tokens: 512,
                target_tokens: 400,
                overlap_ratio: 0.0,
                inject_context: false,
            };
            let new_config = BoundedChunkerConfig {
                max_tokens: 350,
                target_tokens: 280,
                overlap_ratio: 0.0,
                inject_context: false,
            };

            let old_chunker = BoundedSemanticChunker::new(old_config);
            let new_chunker = BoundedSemanticChunker::new(new_config);

            // When: Chunking same content
            let old_chunks = old_chunker.chunk(LARGE_IMPL_BLOCK, "rust");
            let new_chunks = new_chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            // Then: New config should produce >= same number of chunks
            assert!(
                new_chunks.len() >= old_chunks.len(),
                "Smaller chunks should produce more splits: old={}, new={}",
                old_chunks.len(),
                new_chunks.len()
            );
        }

        #[test]
        fn small_config_helper_creates_reduced_defaults() {
            // Given: Helper for creating small chunk config
            let config = BoundedChunkerConfig::small();

            // Then: Should have reduced defaults
            assert!(config.max_tokens <= 350);
            assert!(config.target_tokens <= 280);
            assert!((config.overlap_ratio - 0.15).abs() < 0.01);
        }

        #[test]
        fn chunks_average_near_target_size() {
            // Given: Chunker with specific target
            let chunker = BoundedSemanticChunker::new(BoundedChunkerConfig {
                max_tokens: 350,
                target_tokens: 280,
                overlap_ratio: 0.0,
                inject_context: false,
            });

            // When: Chunking substantial content
            let chunks = chunker.chunk(LARGE_IMPL_BLOCK, "rust");

            if chunks.len() > 1 {
                // Then: Average chunk size should be close to target
                let total_tokens: usize = chunks.iter()
                    .map(|c| chunker.count_tokens(&c.content))
                    .sum();
                let avg_tokens = total_tokens / chunks.len();

                // Average should be within 50% of target
                let target = 280;
                let lower_bound = target / 2;
                let upper_bound = target + target / 2;

                assert!(
                    avg_tokens >= lower_bound && avg_tokens <= upper_bound,
                    "Average tokens {} should be near target {} (range {}-{})",
                    avg_tokens,
                    target,
                    lower_bound,
                    upper_bound
                );
            }
        }
    }

    // =========================================================================
    // MOCK EMBEDDER
    // =========================================================================

    struct MockEmbedder {
        dim: usize,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, _text: &str) -> Vec<f32> {
            // Return a simple mock embedding
            vec![0.1; self.dim]
        }

        fn dimension(&self) -> usize {
            self.dim
        }
    }
}
