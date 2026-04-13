#[cfg(test)]
mod tests {
    use crate::specialization::*;

    /// Helper: create a DivHint with the default max (16).
    fn dh(divisor: i32) -> DivHint {
        DivHint { divisor, max: 16 }
    }

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
    fn divhint_from_value() {
        assert_eq!(DivHint::from_value(1024), dh(16));
        assert_eq!(DivHint::from_value(12), dh(4));
        assert_eq!(DivHint::from_value(7), dh(1));
        assert_eq!(DivHint::from_value(0), dh(16));
    }

    #[test]
    fn divhint_with_max() {
        let h: DivHint = DivHint::from_value(1024).with_max(8);
        assert_eq!(h.divisor, 8);
        assert_eq!(h.max, 8);

        let h: DivHint = DivHint::from_value(3).with_max(8);
        assert_eq!(h.divisor, 1); // 3 has divisor 1, still 1 after clamping to 8
    }

    #[test]
    fn divhint_default() {
        let h: DivHint = DivHint::default();
        assert_eq!(h.divisor, 1);
        assert_eq!(h.max, 16);
    }

    #[test]
    fn compute_spec_contiguous_aligned() {
        // 1D tensor: shape=[1024], strides=[1], dtype=f32 (4 bytes), ptr aligned to 16
        let spec = compute_spec(0x1000, &[1024], &[1], 4);
        assert_eq!(spec.shape_div, vec![dh(16)]); // 1024 % 16 == 0
        assert_eq!(spec.stride_div, vec![dh(1)]); // stride=1 in elements
        assert_eq!(spec.stride_one, vec![true]);
        assert_eq!(spec.base_ptr_div, dh(16)); // 0x1000 % 16 == 0
        assert!(spec.elements_disjoint);
    }

    #[test]
    fn compute_spec_2d_row_major() {
        // 2D tensor: shape=[128, 256], strides=[256, 1], dtype=f16 (2 bytes)
        let spec = compute_spec(0x1000, &[128, 256], &[256, 1], 2);
        assert_eq!(spec.shape_div, vec![dh(16), dh(16)]); // both divisible by 16
        assert_eq!(spec.stride_div, vec![dh(16), dh(1)]); // stride=[256,1] in elements
        assert_eq!(spec.stride_one, vec![false, true]);
        assert!(spec.elements_disjoint);
    }

    #[test]
    fn compute_spec_odd_shape() {
        // shape=[1023], strides=[1], dtype=f32
        let spec = compute_spec(0x1000, &[1023], &[1], 4);
        assert_eq!(spec.shape_div, vec![dh(1)]); // 1023 is odd
        assert_eq!(spec.stride_div, vec![dh(1)]); // stride=1 in elements
        assert_eq!(spec.stride_one, vec![true]);
    }

    #[test]
    fn compute_spec_unaligned_ptr() {
        // ptr not aligned to 16
        let spec = compute_spec(0x1004, &[128], &[1], 4);
        assert_eq!(spec.base_ptr_div, dh(4)); // 0x1004 = 4100, divisible by 4
    }

    #[test]
    fn stride_div_is_in_elements_not_bytes() {
        // stride=1 should have div=1 regardless of dtype_bytes.
        // Bug: stride_div was computed as DivHint::from_value(stride * dtype_bytes),
        // giving div=4 for f32 (dtype_bytes=4). Should be div=1 (stride in elements).
        let spec_f32 = compute_spec(0x1000, &[128], &[1], 4);
        assert_eq!(
            spec_f32.stride_div,
            vec![dh(1)],
            "stride=1 with f32: div should be 1 (elements), not 4 (bytes)"
        );

        let spec_f16 = compute_spec(0x1000, &[128], &[1], 2);
        assert_eq!(
            spec_f16.stride_div,
            vec![dh(1)],
            "stride=1 with f16: div should be 1 (elements), not 2 (bytes)"
        );

        // stride=256 should have div=16 regardless of dtype.
        let spec = compute_spec(0x1000, &[128, 256], &[256, 1], 2);
        assert_eq!(
            spec.stride_div,
            vec![dh(16), dh(1)],
            "stride=[256,1]: divs should be [16,1] in elements"
        );
    }

    #[test]
    fn base_ptr_div_is_in_bytes() {
        // base_ptr_div measures raw pointer alignment in bytes, not elements.
        // A pointer at 0x1000 (4096) is 16-byte aligned regardless of dtype.
        // Matches cutile-python: base_addr_divisible_by=16.
        let spec_f32 = compute_spec(0x1000, &[128], &[1], 4);
        assert_eq!(
            spec_f32.base_ptr_div,
            dh(16),
            "0x1000 is 16-byte aligned: base_ptr_div should be 16 (bytes)"
        );

        let spec_f16 = compute_spec(0x1000, &[128], &[1], 2);
        assert_eq!(
            spec_f16.base_ptr_div,
            dh(16),
            "same pointer, different dtype: base_ptr_div is still 16 (bytes, not elements)"
        );

        // 0x1004 = 4100 = 4 * 1025, so 4-byte aligned.
        let spec = compute_spec(0x1004, &[128], &[1], 4);
        assert_eq!(
            spec.base_ptr_div,
            dh(4),
            "0x1004 is 4-byte aligned: base_ptr_div should be 4 (bytes)"
        );

        // 0x1002 = 4098 = 2 * 2049, so 2-byte aligned.
        let spec = compute_spec(0x1002, &[128], &[1], 2);
        assert_eq!(
            spec.base_ptr_div,
            dh(2),
            "0x1002 is 2-byte aligned: base_ptr_div should be 2 (bytes)"
        );
    }

    #[test]
    fn units_elements_vs_bytes() {
        // Combined test: stride_div is in elements, base_ptr_div is in bytes.
        // For a 2D f32 tensor at 0x1000 with shape=[64,128], strides=[128,1]:
        //   shape_div:    [16, 16]  — 64 and 128 are both divisible by 16 (elements)
        //   stride_div:   [16, 1]   — 128 is divisible by 16 (elements), 1 has divisor 1
        //   base_ptr_div: 16        — 0x1000 is 16-byte aligned (bytes)
        //
        // dtype_bytes does NOT affect shape_div or stride_div.
        let spec_f32 = compute_spec(0x1000, &[64, 128], &[128, 1], 4);
        let spec_f16 = compute_spec(0x1000, &[64, 128], &[128, 1], 2);

        assert_eq!(
            spec_f32.shape_div, spec_f16.shape_div,
            "shape_div is in elements: dtype does not affect it"
        );
        assert_eq!(
            spec_f32.stride_div, spec_f16.stride_div,
            "stride_div is in elements: dtype does not affect it"
        );
        assert_eq!(
            spec_f32.base_ptr_div, spec_f16.base_ptr_div,
            "base_ptr_div is in bytes: same pointer → same alignment"
        );
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
