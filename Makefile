# Makefile for Temporal Density-Weighted Topic Model
# Compiles C code with O3 optimization for maximum performance

CC := gcc
CFLAGS := -O3 -Wall -std=c11
CFLAGS_DEBUG := -g -O1 -Wall -std=c11
CFLAGS_ASAN := -fsanitize=address -g -O1 -Wall -std=c11
LDLIBS := -lm

# Targets
TARGET := temporal_lda
TARGET_OMP := temporal_lda_omp
TARGET_DEBUG := temporal_lda_debug
TARGET_ASAN := temporal_lda_asan

# Source files
SOURCES := temporal_lda.c
HEADERS := 

.PHONY: all clean run run-debug test-asan test-repro help

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDLIBS)
	@echo "✓ Built optimized binary: $(TARGET)"

# Parallel build (OpenMP). Same CLI; control threads with OMP_NUM_THREADS.
# Requires a compiler with OpenMP support (gcc; on macOS use gcc or llvm+libomp).
omp: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -fopenmp $(SOURCES) -o $(TARGET_OMP) $(LDLIBS)
	@echo "✓ Built OpenMP binary: $(TARGET_OMP)"

$(TARGET_DEBUG): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS_DEBUG) $(SOURCES) -o $(TARGET_DEBUG) $(LDLIBS)
	@echo "✓ Built debug binary: $(TARGET_DEBUG)"

$(TARGET_ASAN): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS_ASAN) $(SOURCES) -o $(TARGET_ASAN) $(LDLIBS)
	@echo "✓ Built AddressSanitizer binary: $(TARGET_ASAN)"

# Run the optimized version with test data
run: $(TARGET)
	@if [ -d "test_data" ]; then \
		./$(TARGET) --docs test_data/documents.txt \
			--vocab test_data/vocab.txt \
			--metadata test_data/metadata.txt \
			--output results_make --seed 99 --iterations 100; \
	else \
		echo "Error: test_data/ not found. Run 'ruby gen_test.rb' first."; \
		exit 1; \
	fi

# Run with debug symbols
run-debug: $(TARGET_DEBUG)
	@if [ -d "test_data" ]; then \
		./$(TARGET_DEBUG) --docs test_data/documents.txt \
			--vocab test_data/vocab.txt \
			--metadata test_data/metadata.txt \
			--output results_debug --seed 99 --iterations 50; \
	else \
		echo "Error: test_data/ not found. Run 'ruby gen_test.rb' first."; \
		exit 1; \
	fi

# Run with AddressSanitizer for memory checking
test-asan: $(TARGET_ASAN)
	@if [ -d "test_data" ]; then \
		ASAN_OPTIONS=verbosity=0:halt_on_error=1 ./$(TARGET_ASAN) \
			--docs test_data/documents.txt \
			--vocab test_data/vocab.txt \
			--metadata test_data/metadata.txt \
			--output results_asan --seed 99 --iterations 10; \
	else \
		echo "Error: test_data/ not found. Run 'ruby gen_test.rb' first."; \
		exit 1; \
	fi

# The OpenMP sampler promises byte-identical outputs for a fixed seed and
# thread count. Exercise that contract explicitly.
test-repro: omp
	./test_reproducibility.sh ./$(TARGET_OMP)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET_OMP) $(TARGET_DEBUG) $(TARGET_ASAN)
	rm -rf $(TARGET_ASAN).dSYM
	@echo "✓ Cleaned build artifacts"

# Show help
help:
	@echo "Temporal Density-Weighted Topic Model - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all         - Build optimized binary (default)"
	@echo "  make omp         - Build parallel (OpenMP) binary: $(TARGET_OMP)"
	@echo "  make run         - Build and run optimized version with test data"
	@echo "  make clean       - Remove build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make $(TARGET_DEBUG)   - Build debug binary (O1, with symbols)"
	@echo "  make run-debug   - Build and run debug version (50 iterations)"
	@echo "  make $(TARGET_ASAN)  - Build with AddressSanitizer (memory checking)"
	@echo "  make test-asan   - Build and run AddressSanitizer tests"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build optimized binary"
	@echo "  make run          # Run optimized version"
	@echo "  make test-asan    # Check for memory errors"
	@echo "  make test-repro   # Verify fixed-thread OpenMP reproducibility"
	@echo "  make clean        # Clean everything"
