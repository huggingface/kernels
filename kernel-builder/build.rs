fn main() {
    minijinja_embed::embed_templates!("src/pyproject/templates");
    built::write_built_file().expect("Failed to acquire build-time information");
}
