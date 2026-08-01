use std::collections::HashMap;
use std::env;
use std::fs;

/// Application configuration parsed from a `key=value` file.
pub struct Config {
    values: HashMap<String, String>,
}

impl Config {
    /// Load a config file, ignoring blank lines and `#` comments.
    pub fn load(path: &str) -> Config {
        let text = fs::read_to_string(path).unwrap_or_default();
        let mut values = HashMap::new();
        for line in text.lines() {
            if let Some((k, v)) = parse_line(line) {
                values.insert(k, v);
            }
        }
        Config { values }
    }

    /// Fetch a value with a fallback default.
    pub fn get_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.values.get(key).map(String::as_str).unwrap_or(default)
    }
}

/// Split one `key=value` line; returns None for comments/blanks.
fn parse_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (k, v) = line.split_once('=')?;
    Some((k.trim().to_string(), v.trim().to_string()))
}

/// Read an environment variable with a fallback default.
pub fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
