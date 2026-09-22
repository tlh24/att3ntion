#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
// Implemented in the optional sm_90a objects (shared_hopper*.cu). Without them the
// stubs return false, so callers fall back and nothing extra has to link.
#ifdef ATT3NTION_WITH_HOPPER
bool launch_att3_shared_hopper(const void*,const void*,const void*,const void*,const void*,
 void*,float*,float*,const bool*,int,int,int,float,int,cudaStream_t,int);
bool launch_att3_shared_hopper_dq(const void*,const void*,const void*,const void*,const void*,const void*,
 const float*,const float*,const float*,float*,const bool*,int,int,int,int,float,cudaStream_t,const uint8_t*);
#else
inline bool launch_att3_shared_hopper(const void*,const void*,const void*,const void*,const void*,
 void*,float*,float*,const bool*,int,int,int,float,int,cudaStream_t,int){return false;}
inline bool launch_att3_shared_hopper_dq(const void*,const void*,const void*,const void*,const void*,const void*,
 const float*,const float*,const float*,float*,const bool*,int,int,int,int,float,cudaStream_t,const uint8_t*){return false;}
#endif
