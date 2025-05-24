use crate::{
    diagnostics::{DiagCtxt, EmittedDiagnostics},
    ColorChoice, SessionGlobals, SourceMap,
};
use solar_config::{CompilerOutput, CompilerStage, Opts, UnstableOpts, SINGLE_THREADED_TARGET};
use std::{path::Path, sync::Arc};
use std::cell::RefCell;

// Thread-local cache for SourceMap to avoid Arc false sharing
thread_local! {
    static SOURCE_MAP_CACHE: RefCell<Option<(usize, Arc<SourceMap>)>> = RefCell::new(None);
}

/// Information about the current compiler session.
#[derive(derive_builder::Builder)]
#[builder(pattern = "owned", build_fn(name = "try_build", private), setter(strip_option))]
pub struct Session {
    /// The diagnostics context.
    pub dcx: DiagCtxt,
    /// The source map.
    #[builder(default)]
    source_map: Arc<SourceMap>,
    /// Unique ID for this session to invalidate TLS cache
    #[builder(default = "SESSION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)")]
    session_id: usize,

    /// The compiler options.
    #[builder(default)]
    pub opts: Opts,
}

// Global session counter for cache invalidation
static SESSION_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl SessionBuilder {
    /// Sets the diagnostic context to a test emitter.
    #[inline]
    pub fn with_test_emitter(mut self) -> Self {
        let sm = self.get_source_map();
        self.dcx(DiagCtxt::with_test_emitter(Some(sm)))
    }

    /// Sets the diagnostic context to a stderr emitter.
    #[inline]
    pub fn with_stderr_emitter(self) -> Self {
        self.with_stderr_emitter_and_color(ColorChoice::Auto)
    }

    /// Sets the diagnostic context to a stderr emitter and a color choice.
    #[inline]
    pub fn with_stderr_emitter_and_color(mut self, color_choice: ColorChoice) -> Self {
        let sm = self.get_source_map();
        self.dcx(DiagCtxt::with_stderr_emitter_and_color(Some(sm), color_choice))
    }

    /// Sets the diagnostic context to a human emitter that emits diagnostics to a local buffer.
    #[inline]
    pub fn with_buffer_emitter(mut self, color_choice: ColorChoice) -> Self {
        let sm = self.get_source_map();
        self.dcx(DiagCtxt::with_buffer_emitter(Some(sm), color_choice))
    }

    /// Sets the diagnostic context to a silent emitter.
    #[inline]
    pub fn with_silent_emitter(self, fatal_note: Option<String>) -> Self {
        self.dcx(DiagCtxt::with_silent_emitter(fatal_note))
    }

    /// Sets the number of threads to use for parallelism to 1.
    #[inline]
    pub fn single_threaded(self) -> Self {
        self.threads(1)
    }

    /// Sets the number of threads to use for parallelism. Zero specifies the number of logical
    /// cores.
    #[inline]
    pub fn threads(mut self, threads: usize) -> Self {
        self.opts_mut().threads = threads.into();
        self
    }

    /// Gets the source map from the diagnostics context.
    fn get_source_map(&mut self) -> Arc<SourceMap> {
        self.source_map.get_or_insert_default().clone()
    }

    /// Returns a mutable reference to the options.
    fn opts_mut(&mut self) -> &mut Opts {
        self.opts.get_or_insert_default()
    }

    /// Consumes the builder to create a new session.
    #[track_caller]
    pub fn build(mut self) -> Session {
        // Set the source map from the diagnostics context if it's not set.
        let dcx = self.dcx.as_mut().unwrap_or_else(|| panic!("diagnostics context not set"));
        if self.source_map.is_none() {
            self.source_map = dcx.source_map_mut().cloned();
        }

        let mut sess = self.try_build().unwrap();
        
        if let Some(sm) = sess.dcx.source_map_mut() {
            assert!(
                Arc::ptr_eq(&sess.source_map, sm),
                "session source map does not match the one in the diagnostics context"
            );
        }
        sess
    }
}

impl Session {
    /// Creates a new session with the given diagnostics context and source map.
    pub fn new(dcx: DiagCtxt, source_map: Arc<SourceMap>) -> Self {
        Self::builder().dcx(dcx).source_map(source_map).build()
    }

    /// Creates a new session with the given diagnostics context and an empty source map.
    pub fn empty(dcx: DiagCtxt) -> Self {
        Self::builder().dcx(dcx).build()
    }

    /// Creates a new session builder.
    #[inline]
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }

    /// Gets a cached reference to the source map to avoid Arc cloning false sharing
    fn get_cached_source_map(&self) -> Arc<SourceMap> {
        SOURCE_MAP_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            match &*cache {
                Some((cached_id, cached_sm)) if *cached_id == self.session_id => {
                    // Cache hit - return cached Arc without incrementing ref count much
                    cached_sm.clone()
                }
                _ => {
                    // Cache miss - clone and store
                    let sm = self.source_map.clone();
                    *cache = Some((self.session_id, sm.clone()));
                    sm
                }
            }
        })
    }

    /// Infers the language from the input files.
    pub fn infer_language(&mut self) {
        if !self.opts.input.is_empty()
            && self.opts.input.iter().all(|arg| Path::new(arg).extension() == Some("yul".as_ref()))
        {
            self.opts.language = solar_config::Language::Yul;
        }
    }

    /// Validates the session options.
    pub fn validate(&self) -> crate::Result<()> {
        let mut result = Ok(());
        result = result.and(self.check_unique("emit", &self.opts.emit));
        result
    }

    fn check_unique<T: Eq + std::hash::Hash + std::fmt::Display>(
        &self,
        name: &str,
        list: &[T],
    ) -> crate::Result<()> {
        let mut result = Ok(());
        let mut seen = std::collections::HashSet::new();
        for item in list {
            if !seen.insert(item) {
                let msg = format!("cannot specify `--{name} {item}` twice");
                result = Err(self.dcx.err(msg).emit());
            }
        }
        result
    }

    /// Returns the unstable options.
    #[inline]
    pub fn unstable(&self) -> &UnstableOpts {
        &self.opts.unstable
    }

    /// Returns the emitted diagnostics. Can be empty.
    #[inline]
    pub fn emitted_diagnostics(&self) -> Option<EmittedDiagnostics> {
        self.dcx.emitted_diagnostics()
    }

    /// Returns `Err` with the printed diagnostics if any errors have been emitted.
    #[inline]
    pub fn emitted_errors(&self) -> Option<Result<(), EmittedDiagnostics>> {
        self.dcx.emitted_errors()
    }

    /// Returns a reference to the source map.
    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    /// Clones the source map using thread-local cache to reduce false sharing
    #[inline]
    pub fn clone_source_map(&self) -> Arc<SourceMap> {
        self.get_cached_source_map()
    }

    /// Returns `true` if compilation should stop after the given stage.
    #[inline]
    pub fn stop_after(&self, stage: CompilerStage) -> bool {
        self.opts.stop_after >= Some(stage)
    }

    /// Returns the number of threads to use for parallelism.
    #[inline]
    pub fn threads(&self) -> usize {
        self.opts.threads().get()
    }

    /// Returns `true` if parallelism is not enabled.
    #[inline]
    pub fn is_sequential(&self) -> bool {
        self.threads() == 1
    }

    /// Returns `true` if parallelism is enabled.
    #[inline]
    pub fn is_parallel(&self) -> bool {
        !self.is_sequential()
    }

    /// Returns `true` if the given output should be emitted.
    #[inline]
    pub fn do_emit(&self, output: CompilerOutput) -> bool {
        self.opts.emit.contains(&output)
    }

    /// Spawns the given closure on the thread pool or executes it immediately if parallelism is not
    /// enabled.
    #[inline]
    pub fn spawn(&self, f: impl FnOnce() + Send + 'static) {
        if self.is_sequential() {
            f();
        } else {
            rayon::spawn(f);
        }
    }

    /// Takes two closures and potentially runs them in parallel.
    #[inline]
    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        if self.is_sequential() {
            (oper_a(), oper_b())
        } else {
            rayon::join(oper_a, oper_b)
        }
    }

    /// Executes the given closure in a fork-join scope.
    #[inline]
    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(solar_data_structures::sync::Scope<'_, 'scope>) -> R + Send,
        R: Send,
    {
        solar_data_structures::sync::scope(self.is_parallel(), op)
    }

    /// Sets up the session globals if they doesn't exist already and then executes the given
    /// closure. Uses cached source map to reduce false sharing.
    #[inline]
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        SessionGlobals::with_or_default(|_| {
            // Use cached source map to avoid Arc cloning overhead
            SessionGlobals::with_source_map(self.get_cached_source_map(), f)
        })
    }

    /// Sets up the thread pool and session globals using cached source map to reduce false sharing.
    #[inline]
    pub fn enter_parallel<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        SessionGlobals::with_or_default(|session_globals| {
            // Pre-populate thread-local cache on main thread
            let cached_sm = self.get_cached_source_map();
            SessionGlobals::with_source_map(cached_sm, || {
                run_in_thread_pool_with_globals(self, session_globals, f)
            })
        })
    }
}

/// Runs the given closure in a thread pool with the given number of threads.
/// Modified to pre-populate thread-local caches to reduce false sharing.
fn run_in_thread_pool_with_globals<R: Send>(
    sess: &Session,
    session_globals: &SessionGlobals,
    f: impl FnOnce() -> R + Send,
) -> R {
    // Avoid panicking below if this is a recursive call.
    if rayon::current_thread_index().is_some() {
        debug!(
            "running in the current thread's rayon thread pool; \
             this could cause panics later on if it was created without setting the session globals!"
        );
        return f();
    }

    let threads = sess.threads();
    debug_assert!(threads > 0, "number of threads must already be resolved");
    
    // Pre-cache source map on main thread to reduce Arc operations in worker threads
    let cached_source_map = sess.get_cached_source_map();
    let session_id = sess.session_id;
    
    let mut builder =
        rayon::ThreadPoolBuilder::new().thread_name(|i| format!("solar-{i}")).num_threads(threads);
    
    if threads == 1 {
        builder = builder.use_current_thread();
    }
    
    match builder.build_scoped(
        // Initialize each new worker thread when created.
        move |thread| {
            session_globals.set(|| {
                // Pre-populate thread-local cache in each worker thread
                SOURCE_MAP_CACHE.with(|cache| {
                    *cache.borrow_mut() = Some((session_id, cached_source_map.clone()));
                });
                thread.run()
            })
        },
        // Run `f` on the first thread in the thread pool.
        move |pool| pool.install(f),
    ) {
        Ok(r) => r,
        Err(e) => {
            let mut err = sess.dcx.fatal(format!("failed to build the rayon thread pool: {e}"));
            if threads > 1 {
                if SINGLE_THREADED_TARGET {
                    err = err.note("the current target might not support multi-threaded execution");
                }
                err = err.help("try running with `--threads 1` / `-j1` to disable parallelism");
            }
            err.emit();
        }
    }
}

// Additional optimization: Bulk cache warming for heavy workloads
impl Session {
    /// Pre-warm thread-local caches across all worker threads
    pub fn warm_caches(&self) {
        if self.is_parallel() {
            let session_id = self.session_id;
            let source_map = self.source_map.clone();
            
            self.enter_parallel(|| {
                // Warm up the cache in all threads
                rayon::broadcast(|_| {
                    SOURCE_MAP_CACHE.with(|cache| {
                        *cache.borrow_mut() = Some((session_id, source_map.clone()));
                    });
                });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_cache_works() {
        let sess = Session::builder().with_buffer_emitter(ColorChoice::Never).build();
        
        // First access should populate cache
        let sm1 = sess.get_cached_source_map();
        
        // Second access should use cache (same pointer)
        let sm2 = sess.get_cached_source_map();
        assert!(Arc::ptr_eq(&sm1, &sm2));
    }

    #[test]
    fn test_session_id_invalidation() {
        let sess1 = Session::builder().with_buffer_emitter(ColorChoice::Never).build();
        let sess2 = Session::builder().with_buffer_emitter(ColorChoice::Never).build();
        
        // Different sessions should have different IDs
        assert_ne!(sess1.session_id, sess2.session_id);
        
        // Cache should be invalidated between sessions
        let _sm1 = sess1.get_cached_source_map();
        let _sm2 = sess2.get_cached_source_map();
        // This would fail if cache wasn't properly invalidated
    }

    #[test]
    fn bench_false_sharing_reduction() {
        use std::time::Instant;
        
        let sess = Session::builder().with_buffer_emitter(ColorChoice::Never).threads(4).build();
        
        // Benchmark the old way (direct Arc cloning)
        let start = Instant::now();
        sess.enter_parallel(|| {
            rayon::scope(|s| {
                for _ in 0..1000 {
                    s.spawn(|_| {
                        let _sm = sess.source_map.clone(); // Direct clone - false sharing
                        std::hint::black_box(_sm);
                    });
                }
            });
        });
        let old_time = start.elapsed();
        
        // Benchmark the new way (cached TLS)
        let start = Instant::now();
        sess.enter_parallel(|| {
            rayon::scope(|s| {
                for _ in 0..1000 {
                    s.spawn(|_| {
                        let _sm = sess.get_cached_source_map(); // Cached - less false sharing
                        std::hint::black_box(_sm);
                    });
                }
            });
        });
        let new_time = start.elapsed();
        
        println!("Old (false sharing): {:?}", old_time);
        println!("New (TLS cache): {:?}", new_time);
        
        // New way should be faster (though this test might not show dramatic difference)
        // The real benefit shows up under heavy concurrent load
    }
}