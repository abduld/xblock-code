var extra_keywords = '__device__ __global__ __host__ __constant__ ' +
  '__shared__ __inline__ __align__ __thread__' +
  '__import__ __export__ __location__';
var extra_types = 'char1 char2 char3 char4 ' +
  'uchar1 uchar2 uchar3 uchar4 ' +
  'short1 short2 short3 short4' +
  'ushort1 ushort2 ushort3 ushort4 ' +
  'int1 int2 int3 int4 ' +
  'uint1 uint2 uint3 uint4 ' +
  'long1 long2 long3 long4 ' +
  'ulong1 ulong2 ulong3 ulong4 ' +
  'float1 float2 float3 float4 ' +
  'ufloat1 ufloat2 ufloat3 ufloat4 ' +
  'dim3 texture textureReference ' +
  'cudaError_t cudaDeviceProp cudaMemcpyKind';

var words = function(str) {
  var res = {};
  _.each(str.split(' '),
    function(word) {
      res[word] = true;
    }
  );
  return res;
};
var base_mode = CodeMirror.mimeModes["text/x-c++src"];
var cuda_mode = _.extend(
  base_mode, {
    keywords: words(_.keys(base_mode.keywords).join(' ') + extra_keywords),
    types: words(_.keys(base_mode.types).join(' ') + extra_types)
  }
);
CodeMirror.defineMIME("text/x-cuda-src", cuda_mode);
