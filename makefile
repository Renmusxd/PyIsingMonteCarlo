wheel:
	maturin build --release --strip --no-sdist

.PHONY: clean
clean:
	cargo clean
