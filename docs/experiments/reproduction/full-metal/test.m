#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>

static int check(id<MTLDevice> dev, id<MTLComputePipelineState> pipeline, int width, int rows, int tokens, int zero) {
    id<MTLBuffer> wb=[dev newBufferWithLength:width*rows/4+32 options:MTLResourceStorageModeShared];
    id<MTLBuffer> xb=[dev newBufferWithLength:width*tokens*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb=[dev newBufferWithLength:rows*tokens*sizeof(float) options:MTLResourceStorageModeShared];
    unsigned char *w=wb.contents;
    float *x=xb.contents,*y=yb.contents;
    memset(w,0,width*rows/4+32);
    for(int r=0;r<rows;r++)for(int k=0;k<width;k++){
        int code=(k*7+r*11+k/17)%3;
        w[r*width/4+(k/128)*32+k%32]|=code<<(6-2*((k%128)/32));
    }
    *(float *)(w+width*rows/4)=0.5f;
    for(int i=0;i<width*tokens;i++)x[i]=zero?0.0f:(i%257-128)*0.015625f;
    uint64_t shape[4]={width,rows,tokens,0},strides[2]={width*sizeof(float),rows*sizeof(float)};
    id<MTLCommandQueue> queue=[dev newCommandQueue];
    id<MTLCommandBuffer> cmd=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBytes:shape length:sizeof(shape) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];[enc setBuffer:xb offset:0 atIndex:2];[enc setBuffer:yb offset:0 atIndex:3];
    [enc setBytes:strides length:sizeof(strides) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((rows+3)/4,tokens,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cmd.error.description.UTF8String);return 1;}
    float maxdiff=0;
    for(int t=0;t<tokens;t++){
        float amax=0;for(int k=0;k<width;k++)amax=fmaxf(amax,fabsf(x[t*width+k]));
        float scale=amax>0?127.0f/amax:1.0f;
        for(int r=0;r<rows;r++){
            int sum=0;for(int k=0;k<width;k++)sum+=((k*7+r*11+k/17)%3-1)*(int)roundf(x[t*width+k]*scale);
            float ref=sum*(0.5f/scale),diff=fabsf(ref-y[t*rows+r]);maxdiff=fmaxf(maxdiff,diff);
            if(diff>1e-5f){fprintf(stderr,"Mismatch r%d t%d ref%g gpu%g\n",r,t,ref,y[t*rows+r]);return 2;}
        }
    }
    printf("PASS I2_S shader width%d rows%d tokens%d zero%d maxdiff%g\n",width,rows,tokens,zero,maxdiff);
    return 0;
}

int main(int argc,char **argv){@autoreleasepool{
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();if(!dev)return 3;
    NSError *error=nil;
    NSString *body=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    NSString *source=[@"#include <metal_stdlib>\nusing namespace metal;\n" stringByAppendingString:body];
    id<MTLLibrary> lib=[dev newLibraryWithSource:source options:nil error:&error];
    if(!lib){fprintf(stderr,"%s\n",error.description.UTF8String);return 4;}
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mv_i2s_reference"] error:&error];
    if(!pipeline)return 5;
    for(int zero=0;zero<2;zero++){if(check(dev,pipeline,128,7,5,zero)||check(dev,pipeline,2560,32,5,zero)||check(dev,pipeline,6912,17,1,zero))return 6;}
    return 0;
}}
