//! Tests for GPU radix sort.

#[cfg(feature = "cuda")]
mod tests {
    use forge_runtime::radix_sort::RadixSorter;
    use forge_runtime::{Array, Device};

    fn skip_if_no_gpu() -> bool {
        forge_runtime::cuda::init();
        forge_runtime::cuda::device_count() == 0
    }

    #[test]
    fn test_radix_sort_small() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let n = 16;
        let keys_data = vec![15u32, 3, 7, 1, 9, 0, 12, 5, 8, 2, 14, 6, 11, 4, 13, 10];
        let vals_data: Vec<u32> = (0..n as u32).collect();

        let mut keys = Array::from_vec(keys_data.clone(), Device::Cuda(0));
        let mut vals = Array::from_vec(vals_data.clone(), Device::Cuda(0));

        let mut sorter = RadixSorter::new(n, 0);
        sorter.sort(&mut keys, &mut vals, n, 4).expect("sort failed");

        let sorted_keys = keys.to_vec();
        let sorted_vals = vals.to_vec();

        // Keys should be sorted
        for i in 1..n {
            assert!(sorted_keys[i] >= sorted_keys[i - 1],
                "Keys not sorted at {}: {} > {}", i, sorted_keys[i-1], sorted_keys[i]);
        }

        // Values should be the permutation corresponding to the sort
        for i in 0..n {
            assert_eq!(keys_data[sorted_vals[i] as usize], sorted_keys[i],
                "Key-value mismatch at {}", i);
        }

        eprintln!("✅ Radix sort small: sorted {} elements correctly", n);
        eprintln!("  Keys: {:?}", sorted_keys);
        eprintln!("  Vals: {:?}", sorted_vals);
    }

    #[test]
    fn test_radix_sort_medium() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let n = 10_000;
        // Random-ish keys using a simple PRNG
        let mut keys_data = Vec::with_capacity(n);
        let mut state = 12345u32;
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            keys_data.push(state & 0xFFFFF); // 20-bit keys
        }
        let vals_data: Vec<u32> = (0..n as u32).collect();

        let mut keys = Array::from_vec(keys_data.clone(), Device::Cuda(0));
        let mut vals = Array::from_vec(vals_data.clone(), Device::Cuda(0));

        let mut sorter = RadixSorter::new(n, 0);
        sorter.sort(&mut keys, &mut vals, n, 20).expect("sort failed");

        let sorted_keys = keys.to_vec();
        let sorted_vals = vals.to_vec();

        // Verify sorted
        for i in 1..n {
            assert!(sorted_keys[i] >= sorted_keys[i - 1],
                "Keys not sorted at {}: {} > {}", i, sorted_keys[i-1], sorted_keys[i]);
        }

        // Verify key-value correspondence
        for i in 0..n {
            assert_eq!(keys_data[sorted_vals[i] as usize], sorted_keys[i],
                "Key-value mismatch at {}", i);
        }

        eprintln!("✅ Radix sort medium: sorted {} elements correctly", n);
    }

    #[test]
    fn test_radix_sort_compare_cpu_reference() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let n = 5000;
        let mut keys_data = Vec::with_capacity(n);
        let mut state = 42u32;
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            keys_data.push(state & 0xFFFF); // 16-bit keys
        }
        let vals_data: Vec<u32> = (0..n as u32).collect();

        // CPU reference sort
        let mut cpu_pairs: Vec<(u32, u32)> = keys_data.iter().cloned().zip(vals_data.iter().cloned()).collect();
        cpu_pairs.sort_by_key(|&(k, _)| k);

        // GPU radix sort
        let mut keys = Array::from_vec(keys_data.clone(), Device::Cuda(0));
        let mut vals = Array::from_vec(vals_data.clone(), Device::Cuda(0));

        let mut sorter = RadixSorter::new(n, 0);
        sorter.sort(&mut keys, &mut vals, n, 16).expect("sort failed");

        let gpu_keys = keys.to_vec();
        let gpu_vals = vals.to_vec();

        // Compare sorted keys
        let cpu_keys: Vec<u32> = cpu_pairs.iter().map(|&(k, _)| k).collect();
        assert_eq!(gpu_keys, cpu_keys, "GPU sorted keys don't match CPU reference");

        // Verify value correspondence (stability not guaranteed, but keys must match)
        for i in 0..n {
            assert_eq!(keys_data[gpu_vals[i] as usize], gpu_keys[i],
                "Key-value mismatch at {}", i);
        }

        eprintln!("✅ Radix sort matches CPU reference for {} elements", n);
    }

    #[test]
    fn test_find_cell_boundaries() {
        if skip_if_no_gpu() {
            eprintln!("Skipping: no GPU");
            return;
        }

        let num_cells = 8;
        // Sorted cell IDs: 0,0,1,1,1,3,3,5
        let sorted_cells = vec![0u32, 0, 1, 1, 1, 3, 3, 5];
        let n = sorted_cells.len();

        let cell_ids = Array::from_vec(sorted_cells, Device::Cuda(0));
        let mut cell_start = Array::<u32>::zeros(num_cells, Device::Cuda(0));
        let mut cell_end = Array::<u32>::zeros(num_cells, Device::Cuda(0));

        forge_runtime::radix_sort::find_cell_boundaries(
            &cell_ids, &mut cell_start, &mut cell_end, n, num_cells, 0
        ).expect("find_cell_boundaries failed");

        let starts = cell_start.to_vec();
        let ends = cell_end.to_vec();

        assert_eq!(starts[0], 0); assert_eq!(ends[0], 2);  // cell 0: [0,2)
        assert_eq!(starts[1], 2); assert_eq!(ends[1], 5);  // cell 1: [2,5)
        assert_eq!(starts[3], 5); assert_eq!(ends[3], 7);  // cell 3: [5,7)
        assert_eq!(starts[5], 7); assert_eq!(ends[5], 8);  // cell 5: [7,8)

        // Cells 2, 4, 6, 7 should have sentinel start (0xFFFFFFFF) and 0 end
        assert_eq!(starts[2], 0xFFFFFFFF);
        assert_eq!(ends[2], 0);

        eprintln!("✅ find_cell_boundaries: correct boundaries detected");
    }
}
