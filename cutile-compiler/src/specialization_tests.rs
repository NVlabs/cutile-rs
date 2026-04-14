#[cfg(test)]
mod tests {
    use crate::specialization::*;

    #[test]
    fn max_pow2_divisor_powers_of_two() {
        assert_eq!(max_pow2_divisor(1), 1);
        assert_eq!(max_pow2_divisor(2), 2);
        assert_eq!(max_pow2_divisor(4), 4);
        assert_eq!(max_pow2_divisor(8), 8);
        assert_eq!(max_pow2_divisor(16), 16);
        // Clamped to 16.
        assert_eq!(max_pow2_divisor(32), 16);
        assert_eq!(max_pow2_divisor(1024), 16);
    }

    #[test]
    fn max_pow2_divisor_non_powers() {
        assert_eq!(max_pow2_divisor(3), 1); // odd
        assert_eq!(max_pow2_divisor(6), 2); // 2 * 3
        assert_eq!(max_pow2_divisor(12), 4); // 4 * 3
        assert_eq!(max_pow2_divisor(24), 8); // 8 * 3
        assert_eq!(max_pow2_divisor(48), 16); // 16 * 3, clamped
        assert_eq!(max_pow2_divisor(7), 1);
        assert_eq!(max_pow2_divisor(1023), 1);
    }

    #[test]
    fn max_pow2_divisor_zero() {
        assert_eq!(max_pow2_divisor(0), 16);
    }

    #[test]
    fn compute_spec_contiguous_aligned() {
        // 1D tensor: shape=[1024], strides=[1], dtype=f32 (4 bytes), ptr aligned to 16
        let spec = compute_spec(0x1000, &[1024], &[1], 4);
        assert_eq!(spec.shape_div, vec![16]); // 1024 % 16 == 0
        assert_eq!(spec.stride_div, vec![4]); // stride_bytes = 1*4 = 4
        assert_eq!(spec.stride_one, vec![true]);
        assert_eq!(spec.base_ptr_div, 16); // 0x1000 % 16 == 0
        assert!(spec.elements_disjoint);
    }

    #[test]
    fn compute_spec_2d_row_major() {
        // 2D tensor: shape=[128, 256], strides=[256, 1], dtype=f16 (2 bytes)
        let spec = compute_spec(0x1000, &[128, 256], &[256, 1], 2);
        assert_eq!(spec.shape_div, vec![16, 16]); // both divisible by 16
        assert_eq!(spec.stride_div, vec![16, 2]); // 256*2=512 -> 16; 1*2=2 -> 2
        assert_eq!(spec.stride_one, vec![false, true]);
        assert!(spec.elements_disjoint);
    }

    #[test]
    fn compute_spec_odd_shape() {
        // shape=[1023], strides=[1], dtype=f32
        let spec = compute_spec(0x1000, &[1023], &[1], 4);
        assert_eq!(spec.shape_div, vec![1]); // 1023 is odd
        assert_eq!(spec.stride_div, vec![4]);
        assert_eq!(spec.stride_one, vec![true]);
    }

    #[test]
    fn compute_spec_unaligned_ptr() {
        // ptr not aligned to 16
        let spec = compute_spec(0x1004, &[128], &[1], 4);
        assert_eq!(spec.base_ptr_div, 4); // 0x1004 = 4100, divisible by 4
    }

    #[test]
    fn compute_spec_disjoint_detection() {
        // Contiguous: disjoint
        let spec = compute_spec(0x1000, &[4, 8], &[8, 1], 4);
        assert!(spec.elements_disjoint);

        // Overlapping: stride[0]=4 < shape[1]*stride[1] = 8*1 = 8
        let spec = compute_spec(0x1000, &[4, 8], &[4, 1], 4);
        assert!(!spec.elements_disjoint);
    }

    #[test]
    fn spec_equality_and_hash() {
        use std::collections::HashSet;
        let a = compute_spec(0x1000, &[128], &[1], 4);
        let b = compute_spec(0x1000, &[128], &[1], 4);
        let c = compute_spec(0x1000, &[127], &[1], 4); // different shape
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }
}
