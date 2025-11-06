fn main() {
    // Allow kani cfg attributes for proof code
    println!("cargo::rustc-check-cfg=cfg(kani)");
}
