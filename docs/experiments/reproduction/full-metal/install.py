from pathlib import Path
import json,shlex,subprocess,shutil
work=Path(__file__).parent.resolve()
build=work.parent/'metal-deep/build'
entries=json.loads((build/'compile_commands.json').read_text())
# Rebuild the separate Metal executable before applying local object overlays.
subprocess.run(['cmake','--build',str(build),'--target','llama-cli','-j','5'],stdout=(work/'base-build.log').open('w'),stderr=subprocess.STDOUT,check=True)
for suffix in ['ggml-metal-ops.cpp','ggml-metal-device.m']:
 entry=next(e for e in entries if e['file'].endswith('/'+suffix));src=Path(entry['file']);text=src.read_text()
 if suffix.endswith('ops.cpp'):
  marker='int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {\n    ggml_tensor * op = ctx->node(idx);'
  assert text.count(marker)==1
  text=text.replace(marker,marker+"""
    if (op->src[0]->type == GGML_TYPE_I2_S) {
        GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_matrix(op->src[0]) && ggml_is_matrix(op->src[1]));
        auto pipeline = ggml_metal_library_get_pipeline(ctx->lib, "kernel_mul_mv_i2s_reference");
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_mul_mv_i2s_reference",
                                                           "kernel_mul_mv_i2s_reference", nullptr);
        }
        uint64_t shape[4] = {uint64_t(op->src[0]->ne[0]), uint64_t(op->src[0]->ne[1]), uint64_t(op->src[1]->ne[1]), 0};
        uint64_t strides[2] = {op->src[1]->nb[1], op->nb[1]};
        ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc, shape, sizeof(shape), 0);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op), 3);
        ggml_metal_encoder_set_bytes(ctx->enc, strides, sizeof(strides), 4);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (shape[1] + 3) / 4, shape[2], 1, 128, 1, 1);
        return 1;
    }
""")
 else:
  marker='''        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;'''
  assert text.count(marker)==1
  text=text.replace(marker,'''        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0]->type == GGML_TYPE_I2_S) {
                return op->op == GGML_OP_MUL_MAT && has_simdgroup_reduction &&
                       ggml_is_matrix(op->src[0]) && ggml_is_contiguous(op->src[0]) &&
                       op->src[0]->ne[0] % 128 == 0 && op->src[0]->ne[0] <= 8192 && op->src[1]->type == GGML_TYPE_F32 &&
                       ggml_is_matrix(op->src[1]) && op->src[1]->nb[0] == sizeof(float) &&
                       op->type == GGML_TYPE_F32;
            }
            return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;''')
 copy=work/suffix;copy.write_text(text)
 command=shlex.split(entry['command']);command[command.index(str(src))]=str(copy);command+=['-I',str(src.parent)]
 subprocess.run(command,cwd=entry['directory'],check=True)
entry=next(e for e in entries if e['file'].endswith('/ggml-metal-embed.s'))
asm=Path(entry['file']);metal=asm.with_name('ggml-metal-embed.metal');backup=work/'base-embed.metal'
if not backup.exists():shutil.copy2(metal,backup)
metal.write_text(backup.read_text()+'\n'+(work/'i2s.metal').read_text())
subprocess.run(shlex.split(entry['command']),cwd=entry['directory'],check=True)
subprocess.run(['cmake','--build',str(build),'--target','llama-cli','-j','5'],check=True,stdout=(work/'link.log').open('w'),stderr=subprocess.STDOUT)
shutil.copy2(build/'bin/llama-cli',work/'llama-cli-full-metal')
