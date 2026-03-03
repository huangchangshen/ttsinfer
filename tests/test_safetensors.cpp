// model.safetensors layout:
//
// Offset   Size       Description
// 0x00     8         header_len (uint64 little endian)
// 0x08     n         header JSON (ASCII text)
//
// header JSON:
//
// {
//   "linear.weight": {
//     "dtype": "F32",
//     "shape": [2,3],
//     "data_offsets": [0,24]
//   }
// }
//
// 0x08+n   24 bytes  raw float32 tensor data (row-major)
//
// Tensor linear.weight (shape [2,3], dtype F32):
//
// [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
//
// Byte offset      Value (hex)                 Comment
// 0x00..0x07       18 00 00 00 00 00 00 00     <-- header length = 24 (example, little endian)
// 0x08..0x??       7B 22 6C 69 ... 7D          <-- JSON ASCII: {"linear.weight": {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}
// 0x??..0x??+3     00 00 80 3F                 <-- float 1.0
// 0x??+4..0x??+7   00 00 00 40                 <-- float 2.0
// 0x??+8..0x??+11  00 00 40 40                 <-- float 3.0
// 0x??+12..0x??+15 00 00 80 40                 <-- float 4.0
// 0x??+16..0x??+19 00 00 A0 40                 <-- float 5.0
// 0x??+20..0x??+23 00 00 C0 40                 <-- float 6.0

#include <cassert>
#include <iostream>
#include "safetensors.h"

using namespace ttsinfer;
using namespace ttsinfer::safetensors;

int main() {
    SafeOpen f("model.safetensors");

    assert(f.contains("linear.weight"));

    auto t = f.get_tensor("linear.weight");

    assert(t.dtype() == DType::F32);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);

    const float* p = static_cast<const float*>(t.data());
    std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
    std::cout << p[3] << " " << p[4] << " " << p[5] << std::endl;

    std::cout << "[OK] test_safetensors passed\n";
    return 0;
}