/**
 * JavaScript Code Generation Test
 * Test: Implement async/await API handling with error management
 */

class APIClient {
  constructor(baseURL) {
    this.baseURL = baseURL;
    this.retryAttempts = 3;
    this.retryDelay = 1000;
  }

  async fetchWithRetry(endpoint, options = {}) {
    let lastError;

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers,
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        console.warn(`Attempt ${attempt} failed: ${error.message}`);

        if (attempt < this.retryAttempts) {
          await this.delay(this.retryDelay * attempt);
        }
      }
    }

    throw new Error(`Failed after ${this.retryAttempts} attempts: ${lastError.message}`);
  }

  async get(endpoint) {
    return this.fetchWithRetry(endpoint, { method: 'GET' });
  }

  async post(endpoint, data) {
    return this.fetchWithRetry(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage
const client = new APIClient('https://api.example.com');

async function demonstrateAPI() {
  try {
    console.log('Fetching user data...');
    const userData = await client.get('/users/1');
    console.log('User data:', userData);

    console.log('Creating new post...');
    const newPost = await client.post('/posts', {
      title: 'Test Post',
      body: 'This is a test',
      userId: 1,
    });
    console.log('Created post:', newPost);
  } catch (error) {
    console.error('API operation failed:', error);
  }
}

module.exports = { APIClient };
