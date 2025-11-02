"""
Simplified Transformer Implementation using NumPy
Demonstrates understanding without external dependencies.

This is a minimal, educational implementation showing core concepts.
For production use, see transformer_implementation.py (PyTorch).
"""

import numpy as np
from typing import Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    Args:
        Q: Queries (batch, seq_len, d_k)
        K: Keys (batch, seq_len, d_k)
        V: Values (batch, seq_len, d_v)
        mask: Optional attention mask

    Returns:
        output: Attention output (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Apply attention to values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (lower triangular) mask for decoder.

    Returns:
        mask: (seq_len, seq_len) with 1s on and below diagonal, 0s above
    """
    return np.tril(np.ones((seq_len, seq_len)))


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        pe: (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer normalization.

    Args:
        x: Input (batch, seq_len, d_model)
        eps: Small constant for numerical stability

    Returns:
        normalized: (batch, seq_len, d_model)
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.
    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def feed_forward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    FFN(x) = GELU(xW1)W2

    Args:
        x: Input (batch, seq_len, d_model)
        W1: First weight matrix (d_model, d_ff)
        W2: Second weight matrix (d_ff, d_model)

    Returns:
        output: (batch, seq_len, d_model)
    """
    hidden = gelu(np.matmul(x, W1))
    output = np.matmul(hidden, W2)
    return output


class SimpleTransformerLayer:
    """
    Single transformer decoder layer (simplified).
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights (in practice, use proper initialization)
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        self.W_ff1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_ff2 = np.random.randn(d_ff, d_model) * 0.02

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through transformer layer.

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # 1. Layer norm + self-attention
        x_norm = layer_norm(x)

        Q = np.matmul(x_norm, self.W_q)
        K = np.matmul(x_norm, self.W_k)
        V = np.matmul(x_norm, self.W_v)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = np.matmul(attn_out, self.W_o)

        # Residual connection
        x = x + attn_out

        # 2. Layer norm + feed-forward
        x_norm = layer_norm(x)
        ff_out = feed_forward(x_norm, self.W_ff1, self.W_ff2)

        # Residual connection
        x = x + ff_out

        return x


def test_transformer_components():
    """Test individual transformer components"""
    print("="*70)
    print("TRANSFORMER COMPONENTS TEST (NumPy Implementation)")
    print("="*70)

    # Test 1: Scaled Dot-Product Attention
    print("\n1. Testing Scaled Dot-Product Attention")
    batch_size, seq_len, d_k = 2, 5, 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   V shape: {V.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")

    # Verify attention weights sum to 1
    weights_sum = attn_weights.sum(axis=-1)
    assert np.allclose(weights_sum, 1.0), "Attention weights don't sum to 1"
    print(f"   ✓ Attention weights sum to 1.0")

    assert output.shape == (batch_size, seq_len, d_k), "Output shape mismatch"
    print("   ✅ PASSED")

    # Test 2: Causal Mask
    print("\n2. Testing Causal Mask")
    seq_len = 4
    mask = create_causal_mask(seq_len)

    print(f"   Mask shape: {mask.shape}")
    print(f"   Causal mask (4×4):")
    print(f"   {mask}")

    expected = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ])
    assert np.array_equal(mask, expected), "Causal mask incorrect"
    print("   ✅ PASSED")

    # Test 3: Positional Encoding
    print("\n3. Testing Positional Encoding")
    seq_len, d_model = 10, 16
    pe = positional_encoding(seq_len, d_model)

    print(f"   PE shape: {pe.shape}")
    print(f"   PE[0, :5]: {pe[0, :5]}")
    print(f"   PE[5, :5]: {pe[5, :5]}")

    assert pe.shape == (seq_len, d_model), "PE shape mismatch"
    # Verify sinusoidal pattern
    assert not np.allclose(pe[0], pe[1]), "PE should differ across positions"
    print("   ✅ PASSED")

    # Test 4: Layer Normalization
    print("\n4. Testing Layer Normalization")
    batch_size, seq_len, d_model = 2, 5, 8
    x = np.random.randn(batch_size, seq_len, d_model) * 10  # Large variance

    x_norm = layer_norm(x)

    print(f"   Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"   Output mean: {x_norm.mean(axis=-1).mean():.6f}")
    print(f"   Output std: {x_norm.std(axis=-1).mean():.6f}")

    # Verify normalization (mean ≈ 0, std ≈ 1 per token)
    assert np.allclose(x_norm.mean(axis=-1), 0, atol=1e-5), "Mean not close to 0"
    assert np.allclose(x_norm.std(axis=-1), 1, atol=1e-5), "Std not close to 1"
    print("   ✅ PASSED")

    # Test 5: GELU Activation
    print("\n5. Testing GELU Activation")
    x = np.array([-3, -1, 0, 1, 3])
    y = gelu(x)

    print(f"   Input: {x}")
    print(f"   GELU output: {y}")

    # Verify GELU properties
    assert y[2] == 0, "GELU(0) should be 0"  # GELU(0) = 0
    assert y[4] > y[3] > y[2], "GELU should be monotonically increasing for x > 0"
    print("   ✅ PASSED")

    # Test 6: Complete Transformer Layer
    print("\n6. Testing Complete Transformer Layer")
    batch_size, seq_len, d_model, d_ff = 2, 8, 64, 256

    layer = SimpleTransformerLayer(d_model, d_ff)

    x = np.random.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    output = layer.forward(x, mask)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    assert output.shape == x.shape, "Output shape should match input"
    assert not np.allclose(output, x), "Output should differ from input"
    print("   ✅ PASSED")

    # Test 7: Multi-layer Forward Pass
    print("\n7. Testing Multi-Layer Forward Pass")
    num_layers = 3
    layers = [SimpleTransformerLayer(d_model, d_ff) for _ in range(num_layers)]

    x = np.random.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    # Add positional encoding
    pe = positional_encoding(seq_len, d_model)
    x = x + pe[np.newaxis, :, :]  # Broadcast to batch

    # Forward through all layers
    for i, layer in enumerate(layers):
        x = layer.forward(x, mask)
        print(f"   After layer {i+1}: mean={x.mean():.4f}, std={x.std():.4f}")

    print(f"   Final output shape: {x.shape}")
    print("   ✅ PASSED")

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)

    # Summary
    print("\n📊 Validated Components:")
    print("   ✅ Scaled dot-product attention")
    print("   ✅ Causal masking (for autoregressive generation)")
    print("   ✅ Positional encoding (sinusoidal)")
    print("   ✅ Layer normalization")
    print("   ✅ GELU activation")
    print("   ✅ Feed-forward networks")
    print("   ✅ Residual connections")
    print("   ✅ Complete transformer layer")
    print("   ✅ Multi-layer stacking")

    print("\n✨ Core transformer concepts validated!")


def demonstrate_attention_pattern():
    """Visualize attention pattern"""
    print("\n" + "="*70)
    print("ATTENTION PATTERN DEMONSTRATION")
    print("="*70)

    # Create simple sequence
    seq_len = 5
    d_k = 4

    # Create queries, keys, values
    Q = np.eye(seq_len, d_k)  # Each position looks for itself
    K = np.eye(seq_len, d_k)
    V = np.arange(seq_len)[:, np.newaxis].repeat(d_k, axis=1)  # Values = position indices

    # Compute attention without mask
    print("\n1. Bidirectional Attention (Encoder-style):")
    output_bi, weights_bi = scaled_dot_product_attention(
        Q[np.newaxis], K[np.newaxis], V[np.newaxis]
    )

    print("   Attention weights:")
    print(weights_bi[0].round(3))
    print("   (Each row shows what a query attends to)")

    # Compute attention with causal mask
    print("\n2. Causal Attention (Decoder-style):")
    mask = create_causal_mask(seq_len)
    output_causal, weights_causal = scaled_dot_product_attention(
        Q[np.newaxis], K[np.newaxis], V[np.newaxis], mask[np.newaxis]
    )

    print("   Attention weights (causal):")
    print(weights_causal[0].round(3))
    print("   (Future positions masked out, only attend to past)")

    print("\n   Key observation:")
    print("   - Position 0 only attends to itself (no past)")
    print("   - Position 1 attends to [0, 1]")
    print("   - Position 4 attends to all previous [0, 1, 2, 3, 4]")


if __name__ == "__main__":
    # Run all tests
    test_transformer_components()

    # Demonstrate attention
    demonstrate_attention_pattern()

    print("\n" + "="*70)
    print("TRANSFORMER VALIDATION COMPLETE ✅")
    print("="*70)
    print("\nThis NumPy implementation demonstrates:")
    print("  • Core transformer mathematics")
    print("  • Attention mechanism (encoder & decoder)")
    print("  • Positional encodings")
    print("  • Layer normalization")
    print("  • Complete forward pass")
    print("\nFor production use, see:")
    print("  • transformer_implementation.py (PyTorch)")
    print("  • TRANSFORMER_ARCHITECTURE_ANALYSIS.md (theory)")
