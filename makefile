wheel:
	maturin build --release --strip --no-sdist --rustc-extra-args="-C debug-assertions"

.PHONY: clean
clean:
	cargo clean
