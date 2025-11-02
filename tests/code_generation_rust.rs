/**
 * Rust Code Generation Test
 * Test: Memory-safe concurrent data structures with ownership and lifetimes
 */

use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::collections::HashMap;

// Thread-safe cache with generic key-value pairs
pub struct ThreadSafeCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    data: Arc<RwLock<HashMap<K, V>>>,
    max_size: usize,
}

impl<K, V> ThreadSafeCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let data = self.data.read().unwrap();
        data.get(key).cloned()
    }

    pub fn insert(&self, key: K, value: V) -> Result<(), String> {
        let mut data = self.data.write().unwrap();

        if data.len() >= self.max_size && !data.contains_key(&key) {
            return Err("Cache is full".to_string());
        }

        data.insert(key, value);
        Ok(())
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let mut data = self.data.write().unwrap();
        data.remove(key)
    }

    pub fn len(&self) -> usize {
        let data = self.data.read().unwrap();
        data.len()
    }

    pub fn clone_handle(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            max_size: self.max_size,
        }
    }
}

// Custom Result type with lifetime parameters
pub struct CacheEntry<'a, V> {
    pub value: &'a V,
    pub age: std::time::Duration,
}

// Demonstrate ownership and borrowing
pub fn process_data<T: std::fmt::Display>(data: T) -> String {
    format!("Processed: {}", data)
}

// Example usage with threads
fn main() {
    let cache: ThreadSafeCache<String, i32> = ThreadSafeCache::new(100);

    let mut handles = vec![];

    // Spawn multiple threads accessing the same cache
    for i in 0..5 {
        let cache_clone = cache.clone_handle();

        let handle = thread::spawn(move || {
            for j in 0..10 {
                let key = format!("key_{}_{}", i, j);
                let value = i * 10 + j;

                match cache_clone.insert(key.clone(), value) {
                    Ok(_) => println!("Thread {}: Inserted {} = {}", i, key, value),
                    Err(e) => eprintln!("Thread {}: Error - {}", i, e),
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final cache size: {}", cache.len());

    // Test retrieval
    if let Some(value) = cache.get(&"key_0_0".to_string()) {
        println!("Retrieved: key_0_0 = {}", value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_operations() {
        let cache: ThreadSafeCache<i32, String> = ThreadSafeCache::new(10);

        cache.insert(1, "one".to_string()).unwrap();
        assert_eq!(cache.get(&1), Some("one".to_string()));

        cache.remove(&1);
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_cache_max_size() {
        let cache: ThreadSafeCache<i32, i32> = ThreadSafeCache::new(2);

        cache.insert(1, 100).unwrap();
        cache.insert(2, 200).unwrap();

        let result = cache.insert(3, 300);
        assert!(result.is_err());
    }
}
