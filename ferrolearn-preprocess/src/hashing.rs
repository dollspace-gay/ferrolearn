//! Shared MurmurHash3 helpers for feature hashing.

pub(crate) fn signed_hash_index(hash: i32, n_features: usize) -> usize {
    let magnitude = u64::from(hash.unsigned_abs());
    (magnitude % n_features as u64) as usize
}

pub(crate) fn murmurhash3_32_signed(bytes: &[u8], seed: u32) -> i32 {
    let hash = murmurhash3_32_u32(bytes, seed);
    hash as i32
}

pub(crate) fn murmurhash3_32_u32(bytes: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    let mut h1 = seed;
    let nblocks = bytes.len() / 4;

    for i in 0..nblocks {
        let offset = i * 4;
        let mut k1 = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);

        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    let tail = &bytes[nblocks * 4..];
    let mut k1 = 0_u32;
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
        }
        1 => {
            k1 ^= tail[0] as u32;
        }
        _ => {}
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= bytes.len() as u32;
    fmix32(h1)
}

fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murmurhash3_matches_sklearn_smoke_values() {
        assert_eq!(murmurhash3_32_signed(b"foo", 0), -156908512);
        assert_eq!(murmurhash3_32_signed(b"foo", 42), -1322301282);
        assert_eq!(murmurhash3_32_u32(b"foo", 0), 4138058784);
        assert_eq!(murmurhash3_32_u32(b"foo", 42), 2972666014);
        assert_eq!(signed_hash_index(-1132748958, 8), 6);
    }
}
