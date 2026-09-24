import torch, torch_npu
p = torch.npu.get_device_properties(0)
for a in ("name","multi_processor_count","cube_core_num","vector_core_num",
          "max_threads_per_multi_processor","warp_size","L2_cache_size"):
    print("  %-30s %s" % (a, getattr(p, a, "(无此属性)")))
