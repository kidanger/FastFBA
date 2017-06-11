#include <cstdio>
#include <vector>
#include <algorithm>

#include <boost/compute.hpp>
#include <boost/compute/types/complex.hpp>

#include <clFFT.h>

#include <ceres/ceres.h>

#include "image.hpp"
#include "vec2.hpp"

namespace compute = boost::compute;
using boost::compute::dim;

using homography_t = std::array<double, 9>;

template <typename T>
void hann(img_t<T>& out, int w, int h, int d=1) {
    out.resize(w, h, d);
    out.set_value(0);
    // with modifications from "Burst photography for high dynamic range and low-light imaging on mobile cameras"
    // namely: half pixel offset and /w|/h instead of /(w-1)|/(h-1)
    for (int l = 0; l < d; l++) {
        for (int y = 0; y < h; y++) {
            T vy = 0.5f * (1 - std::cos(2*M_PI*(y+0.5) / h));
            for (int x = 0; x < w; x++) {
                T vx = 0.5 * (1 - std::cos(2*M_PI*(x+0.5) / w));
                out(x, y, l) = vx * vy;
            }
        }
    }
}

template <typename T>
vec2<T> homography_apply(const homography_t& H, vec2<T> x) {
    T X = H[0]*x[0] + H[1]*x[1] + H[2];
    T Y = H[3]*x[0] + H[4]*x[1] + H[5];
    T Z = H[6]*x[0] + H[7]*x[1] + H[8];
    return { X / Z, Y / Z };
}

struct TranslationResidual {
    TranslationResidual(vec2<double> x, vec2<double> y) : x_(x), y_(y) {}

    template <typename T> bool operator()(const T* const h,
                                          T* residual) const {
        T X = h[0]*x_[0] + h[1]*x_[1] + h[2];
        T Y = h[3]*x_[0] + h[4]*x_[1] + h[5];
        T Z = h[6]*x_[0] + h[7]*x_[1] + h[8];

        vec2<T> p = { X / Z, Y / Z };
        residual[0] = y_[0] - p[0];
        residual[1] = y_[1] - p[1];
        return true;
    }

    private:
    const vec2<double> x_;
    const vec2<double> y_;
};

homography_t homography_from_translations_robust(const std::vector<vec2<vec2<double>>>& translations)
{
    homography_t h = {1,0,0, 0,1,0, 0,0,1};
    ceres::Problem problem;
    double* ph = &h[0];
    for (unsigned i = 0; i < translations.size(); i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TranslationResidual, 2, 9>(
                                    new TranslationResidual(translations[i][0], translations[i][1])),
                                 new ceres::SoftLOneLoss(1.0), ph);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (int k = 0; k < 9; k++)
        h[k] /= h[8];
    return h;
}

template <typename T>
img_t<T> img_from_device(const compute::vector<T>& input,
                                 int w, int h, int d,
                                 compute::command_queue& queue) {
    queue.finish();
    img_t<T> out(w, h, d);
    compute::copy(input.begin(), input.end(), out.data.begin(), queue);
    queue.finish();
    return out;
}

compute::vector<float> img_to_device(const img_t<float>& input, compute::command_queue& queue) {
    queue.finish();
    return compute::vector<float>(input.data.begin(), input.data.end(), queue);
}

const char kernels_src[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    typedef float2 cfloat;

    inline cfloat cmult(cfloat a, cfloat b){
        return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }

    inline cfloat cconj(cfloat a){
        return (cfloat)(a.x, -a.y);
    }

    __kernel void extract_tile(__global const float* input,
                               __global float* output,
                               const int ox, const int oy,
                               const int w, const int h)
    {
        int x = get_global_id(0) + ox;
        int y = get_global_id(1) + oy;
        const int dx = get_global_id(0);
        const int dy = get_global_id(1);

        x = max(0, min(x, w-1));
        y = max(0, min(y, h-1));

        output[(dx+dy*W)*3+0] = input[(x+y*w)*3+0];
        output[(dx+dy*W)*3+1] = input[(x+y*w)*3+1];
        output[(dx+dy*W)*3+2] = input[(x+y*w)*3+2];
    }

    __kernel void fulltohalf(__global const float* input,
                             __global float* output)
    {
        const int x = get_global_id(0);
        const int y = get_global_id(1);
        int dx = (x + W/2) % W;
        int dy = (y + W/2) % W;
        if (dx >= W/4 && dx < W*3/4 && dy >= W/4 && dy < W*3/4) {
            output[(x+y*W)*3+0] = input[(dx+dy*W)*3+0];
            output[(x+y*W)*3+1] = input[(dx+dy*W)*3+1];
            output[(x+y*W)*3+2] = input[(dx+dy*W)*3+2];
        } else {
            output[(x+y*W)*3+0] = 0.f;
            output[(x+y*W)*3+1] = 0.f;
            output[(x+y*W)*3+2] = 0.f;
        }
    }

    __kernel void float2complex(__global const float* input,
                                __global cfloat* output)
    {
        const int x = get_global_id(0);

        output[x].x = input[x];
        output[x].y = 0.f;
    }

    __kernel void magnitude(__global const cfloat* input,
                            __global float* out)
    {
        const int x = get_global_id(0);

        out[x] = (fast_length(input[x*3+0])
                + fast_length(input[x*3+1])
                + fast_length(input[x*3+2])) / 3.f;
    }

    // /!\ transposed result
    __kernel void blur(__global const float* in,
                       __global float* out,
                       __global const float* gaussian, int size)
    {
        const int x = get_global_id(0);
        const int y = get_global_id(1);

        float v = 0.;
        for (int i = -size; i <= size; i++) {
            v += in[(x+i+W)%W+W*y] * gaussian[i+size];
        }
        out[y+W*x] = v;
    }

    __kernel void pow_(__global float* buf, float p)
    {
        const int x = get_global_id(0);
        buf[x] = pow(buf[x], p);
    }

    __kernel void crosscorrelation(__global const cfloat* img,
                                   __global const cfloat* ref,
                                   __global cfloat* cc)
    {
        const int x = get_global_id(0);
        cc[x] = (cmult(img[x*3+0], cconj(ref[x*3+0]))
               + cmult(img[x*3+1], cconj(ref[x*3+1]))
               + cmult(img[x*3+2], cconj(ref[x*3+2]))) / 3.f;
    }

    __kernel void l2residuals(__global const cfloat* cc,
                              __global const float* boxfiltered,
                              __global float* D)
    {
        const int x = get_global_id(0);
        D[x] = boxfiltered[x] - 2.f * cc[x].x;
    }

    __kernel void translate(__global const float2* in,
                            __global float2* out,
                            float dx, float dy)
    {
        const int x = get_global_id(0);
        const int y = get_global_id(1);
        const int wx = (x + W / 2) % W - W / 2;
        const int wy = (y + W / 2) % W - W / 2;

        const float d = 2.f * M_PI_F * (wx * dx / W + wy * dy / W);
        const cfloat phase = (cfloat)(cos(d), sin(d));

        out[(x+y*W)*3+0] = cmult(in[(x+y*W)*3+0], phase);
        out[(x+y*W)*3+1] = cmult(in[(x+y*W)*3+1], phase);
        out[(x+y*W)*3+2] = cmult(in[(x+y*W)*3+2], phase);
    }

    __kernel void accumulate(__global const cfloat* tile,
                             __global const float* hann,
                             __global float* image,
                             __global float* image_weight,
                             const int ox, const int oy,
                             const int w, const int h)
    {
        int x = get_global_id(0) + ox;
        int y = get_global_id(1) + oy;
        const int dx = get_global_id(0);
        const int dy = get_global_id(1);
        float weight = hann[dx+dy*W];

        if (x >= 0 && x < w && y >= 0 && y < h) {
            image_weight[x+y*w] += weight;
            image[(x+y*w)*3+0] += tile[(dx+dy*W)*3+0].x * weight;
            image[(x+y*w)*3+1] += tile[(dx+dy*W)*3+1].x * weight;
            image[(x+y*w)*3+2] += tile[(dx+dy*W)*3+2].x * weight;
        }
    }

    __kernel void unweight(__global float* image,
                           __global const float* image_weight)
    {
        int x = get_global_id(0);
        image[x*3+0] /= image_weight[x];
        image[x*3+1] /= image_weight[x];
        image[x*3+2] /= image_weight[x];
    }

    __kernel void cunweight(__global cfloat* image,
                            __global const float* image_weight)
    {
        int x = get_global_id(0);
        image[x*3+0] /= image_weight[x];
        image[x*3+1] /= image_weight[x];
        image[x*3+2] /= image_weight[x];
    }

    __kernel void fba(__global cfloat* accum,
                      __global float* accum_weight,
                      __global const cfloat* tile,
                      __global const float* tile_weight)
    {
        int x = get_global_id(0);
        const float weight = tile_weight[x];

        accum_weight[x] += weight + 1e-6;
        accum[x*3+0].x += tile[x*3+0].x * weight;
        accum[x*3+0].y += tile[x*3+0].y * weight;
        accum[x*3+1].x += tile[x*3+1].x * weight;
        accum[x*3+1].y += tile[x*3+1].y * weight;
        accum[x*3+2].x += tile[x*3+2].x * weight;
        accum[x*3+2].y += tile[x*3+2].y * weight;
    }

    __kernel void sqr(__global const float* in,
                      __global float* out)
    {
        const int x = get_global_id(0);
        out[x] = in[x] * in[x];
    }

    // /!\ unnormalized + transposed output
    __kernel void boxfilter(__global const float* in,
                            __global float* out)
    {
        const int y = get_global_id(0);

        float v = in[y*W];
        for (int x = 1; x <= hw; x++) {
            v += in[x+y*W] + in[(W-x)+y*W];
        }

        out[y] = v;
        for (int x = 1; x <= hw; x++) {
            v += in[(x+hw)+y*W] - in[(W+x-hw-1)+y*W];
            out[y+x*W] = v;
        }
        for (int x = hw + 1; x < W - hw; x++) {
            v += in[(x+hw)+y*W] - in[(x-hw-1)+y*W];
            out[y+x*W] = v;
        }
        for (int x = W - hw; x < W; x++) {
            v += in[(x-W+hw)+y*W] - in[(x-hw-1)+y*W];
            out[y+x*W] = v;
        }
    }

    __kernel void rgb(__global const float* input,
                      __global float* out)
    {
        const int x = get_global_id(0);
        out[x] = (input[x*3+0]
                + input[x*3+1]
                + input[x*3+2]) / 3.f;
    }
);

struct tile {

    ///////////////
    // constants //
    ///////////////

    int x, y;
    bool use_for_estimation;
    img_t<float> src;
    compute::vector<float> f; // full tiles
    compute::vector<float> f_sqr; // full tiles
    compute::vector<float> h; // half tiles
    compute::vector<std::complex<float>> ff; // fourier full tiles
    compute::vector<std::complex<float>> fh; // fourier half tiles
    compute::vector<float> boxfiltered; // W*W*1
    compute::vector<float> boxfiltered2; // W*W*1
    compute::vector<float> w; // W*W*1
    compute::vector<float> magn; // W*W*1
    compute::vector<float> wblur; // W*W*1
    compute::vector<char> tmpbuf;
    compute::vector<char> tmpbufgray;

    ///////////////////
    // time-variable //
    ///////////////////

    bool valid;
    float dx, dy;
    compute::vector<std::complex<float>> cc; // W*W*1
    compute::vector<float> l2residuals; // W*W*1
    compute::vector<std::complex<float>> rff; // registered fourier full tiles
};

struct image {

    ///////////////
    // constants //
    ///////////////

    int w, h, d, nt;
    img_t<float> src;
    compute::vector<float> dev; // w * h * d
    std::vector<tile> tiles; // nt tiles

    ///////////////////
    // time-variable //
    ///////////////////

    bool allocated;
};

struct result_tile {
    int x, y;

    compute::vector<std::complex<float>> rf;
    compute::vector<std::complex<float>> accum;
    compute::vector<float> accum_weight;
    compute::vector<char> tmpbuf;
};

struct result {
    int w, h, d, nt;
    std::vector<result_tile> tiles;

    compute::vector<float> accumulated; // w * h * d
    compute::vector<float> accumulated_weight; // w * h
};

struct things {
    int W;
    int O;
    float p;

    compute::vector<float> hann;
    compute::vector<float> gaussian;

    compute::command_queue queue;
    clfftPlanHandle ftplan;
    clfftPlanHandle ftplangray;
    compute::program prog;
};

void to_tiles_with_allocation(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("extract_tile");

    for (int y = -T.O; y < image.h; y+=T.O) {
        for (int x = -T.O; x < image.w; x+=T.O) {
            image.tiles.push_back(tile());
            tile& t = image.tiles[image.tiles.size()-1];
            t.x = x;
            t.y = y;
            t.f = compute::vector<float>(T.W * T.W * image.d, ctx);
            kernel.set_args(image.dev, t.f, x, y, image.w, image.h);
            compute::extents<2> offset = dim(0, 0);
            compute::extents<2> ts = dim(T.W, T.W);
            T.queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), ts.data(), 0);

            t.use_for_estimation  = !(t.x+T.W/2 < T.W/2 || t.x+T.W/2 > image.w - T.W/2);
            t.use_for_estimation &= !(t.y+T.W/2 < T.W/2 || t.y+T.W/2 > image.h - T.W/2);
        }
    }
}

void to_tiles_without_allocation(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("extract_tile");

    compute::extents<2> offset = dim(0, 0);
    compute::extents<2> ts = dim(T.W, T.W);
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        kernel.set_args(image.dev, t.f, t.x, t.y, image.w, image.h);
        T.queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), ts.data(), 0);
    }
}

void fulltohalf(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("fulltohalf");

    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        kernel.set_args(t.f, t.h);
        compute::extents<2> offset = dim(0, 0);
        compute::extents<2> ts = dim(T.W, T.W);
        T.queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), ts.data(), 0);
    }
}

void fftfull(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("float2complex");

    cl_command_queue q = T.queue;
    int err;
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];

        // XXX: this line is necessary even though I don't understand why
        t.ff = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);

        kernel.set_args(t.f, t.ff);
        T.queue.enqueue_1d_range_kernel(kernel, 0, t.ff.size(), 0);

        err = clfftEnqueueTransform(T.ftplan, CLFFT_FORWARD, 1, &q, 0, NULL, NULL,
                                    &t.ff.get_buffer().get(), NULL, t.tmpbuf.get_buffer().get());
        assert(!err);
    }
}

void ffthalf(image& image, things& T)
{
    static auto kernel = T.prog.create_kernel("float2complex");

    cl_command_queue q = T.queue;
    int err;
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        kernel.set_args(t.h, t.fh);
        T.queue.enqueue_1d_range_kernel(kernel, 0, t.h.size(), 0);

        err = clfftEnqueueTransform(T.ftplan, CLFFT_FORWARD, 1, &q, 0, NULL, NULL,
                                    &t.fh.get_buffer().get(), NULL, t.tmpbuf.get_buffer().get());
        assert(!err);
    }
}

void backtospace(result& image, things& T)
{
    auto ctx = T.queue.get_context();

    cl_command_queue q = T.queue;
    int err;
    for (int i = 0; i < image.nt; i++) {
        result_tile& t = image.tiles[i];

        compute::copy(t.accum.begin(), t.accum.end(), t.rf.begin(), T.queue);

        err = clfftEnqueueTransform(T.ftplan, CLFFT_BACKWARD, 1, &q, 0, NULL, NULL,
                                    &t.rf.get_buffer().get(), NULL, t.tmpbuf.get_buffer().get());
        assert(!err);
    }
}


void weight(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel_magn = T.prog.create_kernel("magnitude");
    static auto kernel_pow = T.prog.create_kernel("pow_");
    static auto kernel_blur = T.prog.create_kernel("blur");

    compute::extents<2> offset = dim(0, 0);
    compute::extents<2> ts = dim(T.W, T.W);
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];

        kernel_magn.set_args(t.ff, t.magn);
        T.queue.enqueue_1d_range_kernel(kernel_magn, 0, t.magn.size(), 0);

        kernel_blur.set_args(t.magn, t.wblur, T.gaussian, (int)T.gaussian.size()/2);
        T.queue.enqueue_nd_range_kernel(kernel_blur, 2, offset.data(), ts.data(), 0);

        kernel_blur.set_args(t.wblur, t.w, T.gaussian, (int)T.gaussian.size()/2);
        T.queue.enqueue_nd_range_kernel(kernel_blur, 2, offset.data(), ts.data(), 0);

        kernel_pow.set_args(t.w, T.p);
        T.queue.enqueue_1d_range_kernel(kernel_pow, 0, t.w.size(), 0);
    }
}

void boxfilter(image& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel_box = T.prog.create_kernel("boxfilter");
    static auto kernel_sqr = T.prog.create_kernel("sqr");
    static auto kernel_rgb = T.prog.create_kernel("rgb");

    compute::extents<2> offset = dim(0, 0);
    compute::extents<2> ts = dim(T.W, T.W);
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        kernel_sqr.set_args(t.f, t.f_sqr);
        T.queue.enqueue_1d_range_kernel(kernel_sqr, 0, t.f_sqr.size(), 0);

        kernel_rgb.set_args(t.f_sqr, t.boxfiltered);
        T.queue.enqueue_1d_range_kernel(kernel_rgb, 0, t.boxfiltered.size(), 0);

        kernel_box.set_args(t.boxfiltered, t.boxfiltered2);
        T.queue.enqueue_1d_range_kernel(kernel_box, 0, T.W, 0);

        kernel_box.set_args(t.boxfiltered2, t.boxfiltered);
        T.queue.enqueue_1d_range_kernel(kernel_box, 0, T.W, 0);
    }
}

void l2residuals(struct image& image, const struct image& ref, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel_cc = T.prog.create_kernel("crosscorrelation");
    static auto kernel = T.prog.create_kernel("l2residuals");

    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        const tile& tref = ref.tiles[i];
        if (!t.use_for_estimation)
            continue;

        kernel_cc.set_args(t.ff, tref.fh, t.cc);
        T.queue.enqueue_1d_range_kernel(kernel_cc, 0, t.cc.size(), 0);
    }

    cl_command_queue q = T.queue;
    int err;
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        err = clfftEnqueueTransform(T.ftplangray, CLFFT_BACKWARD, 1, &q, 0, NULL, NULL,
                                    &t.cc.get_buffer().get(), NULL, t.tmpbufgray.get_buffer().get());
        assert(!err);
    }

    T.queue.finish();

    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        kernel.set_args(t.cc, t.boxfiltered, t.l2residuals);
        T.queue.enqueue_1d_range_kernel(kernel, 0, t.l2residuals.size(), 0);
    }
}

void fetch_translations(struct image& image, things& T)
{
    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];
        if (!t.use_for_estimation)
            continue;

        auto it = compute::min_element(t.l2residuals.begin(), t.l2residuals.end(), T.queue);
        int x = std::distance(t.l2residuals.begin(), it);
        t.dx = T.W/2 - x % T.W;
        t.dy = T.W/2 - x / T.W;
        t.dx = -t.dx;
        t.dy = -t.dy;
    }
}

void homshift(struct image& image, homography_t hom, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("translate");

    for (int i = 0; i < image.nt; i++) {
        tile& t = image.tiles[i];

        vec2<float> p = {t.x + T.W/2.f, t.y + T.W/2.f};
        vec2<float> d = homography_apply(hom, p) - p;
        d[0] = std::round(d[0]);
        d[1] = std::round(d[1]);

        if (std::abs(d[0]) > T.W/4 || std::abs(d[1]) > T.W/4) {
            t.valid = false;
            continue;
        }
        t.valid = true;

        kernel.set_args(t.ff, t.rff, d[0], d[1]);
        compute::extents<2> offset = dim(0, 0);
        compute::extents<2> ts = dim(T.W, T.W);
        T.queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), ts.data(), 0);
    }

    T.queue.finish();
}

void accumulate(result& image, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("accumulate");

    auto barrier = T.queue.enqueue_marker();

    compute::fill(image.accumulated.begin(), image.accumulated.end(), 0.f, T.queue);
    compute::fill(image.accumulated_weight.begin(), image.accumulated_weight.end(), 0.f, T.queue);

    barrier.wait(); // wait for ifft to finish

    for (int i = 0; i < image.nt; i++) {
        result_tile& t = image.tiles[i];
        int x = t.x;
        int y = t.y;

        kernel.set_args(t.rf, T.hann, image.accumulated, image.accumulated_weight, x, y, image.w, image.h);
        compute::extents<2> offset = dim(0, 0);
        compute::extents<2> ts = dim(T.W, T.W);
        T.queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), ts.data(), 0);
    }

    static auto kernel_unweight = T.prog.create_kernel("unweight");
    kernel_unweight.set_args(image.accumulated, image.accumulated_weight);
    T.queue.enqueue_1d_range_kernel(kernel_unweight, 0, image.accumulated_weight.size(), 0);
}

void fba(result& result, std::vector<image*>& images, things& T)
{
    auto ctx = T.queue.get_context();
    static auto kernel = T.prog.create_kernel("fba");
    static auto kernel_unweight = T.prog.create_kernel("cunweight");

    for (int i = 0; i < result.nt; i++) {
        result_tile& tbuf = result.tiles[i];
        compute::fill(tbuf.accum.begin(), tbuf.accum.end(), 0, T.queue);
        compute::fill(tbuf.accum_weight.begin(), tbuf.accum_weight.end(), 0, T.queue);
    }

    for (int j = 0; j < images.size(); j++) {
        image& image = *images[j];
        for (int i = 0; i < image.nt; i++) {
            result_tile& tbuf = result.tiles[i];
            tile& t = image.tiles[i];

            if (!t.valid)  {
                continue;
            }

            kernel.set_args(tbuf.accum, tbuf.accum_weight, t.rff, t.w);
            T.queue.enqueue_1d_range_kernel(kernel, 0, tbuf.accum_weight.size(), 0);
        }
    }

    for (int i = 0; i < result.nt; i++) {
        result_tile& tbuf = result.tiles[i];
        kernel_unweight.set_args(tbuf.accum, tbuf.accum_weight);
        T.queue.enqueue_1d_range_kernel(kernel_unweight, 0, tbuf.accum_weight.size(), 0);
    }
}

void register_all(image& ref, std::vector<image*>& images, things& T)
{
    for (unsigned i = 0; i < images.size(); i++) {
        if (images[i] != &ref)
            l2residuals(*images[i], ref, T);
    }
    for (unsigned i = 0; i < images.size(); i++) {
        if (images[i] != &ref)
            fetch_translations(*images[i], T);
    }
    for (unsigned i = 0; i < images.size(); i++) {
        if (images[i] == &ref)
            continue;

        int W = T.W;
        std::vector<vec2<vec2<double>>> translations;
        for (int j = 0; j < images[i]->nt; j++) {
            tile& t = images[i]->tiles[j];
            if (!t.use_for_estimation)
                continue;
            vec2<vec2<double>> tr;
            tr[0] = vec2<double>(t.x + T.W/2, t.y + T.W/2);
            tr[1] = vec2<double>(t.x + T.W/2 + t.dx, t.y + T.W/2 + t.dy);
            translations.push_back(tr);
        }

        homography_t H = homography_from_translations_robust(translations);
        homshift(*images[i], H, T);
    }

    for (int j = 0; j < ref.nt; j++) {
        tile& t = ref.tiles[j];
        t.valid = true;
        compute::copy_async(t.ff.begin(), t.ff.end(), t.rff.begin(), T.queue);
    }
}

void fuse_all(result& result, std::vector<image*>& images, things& T)
{
    fba(result, images, T);
    backtospace(result, T);
    accumulate(result, T);
}

void initialize_result(result& result, const image& image, things& T)
{
    auto ctx = T.queue.get_context();

    result.w = image.w;
    result.h = image.h;
    result.d = image.d;
    result.nt = image.nt;
    result.accumulated = compute::vector<float>(image.w*image.h*image.d, ctx);
    result.accumulated_weight = compute::vector<float>(image.w*image.h, ctx);

    result.tiles.resize(image.nt);
    for (int t = 0; t < image.nt; t++) {
        auto& tt = result.tiles[t];
        auto& ti = image.tiles[t];

        tt.x = ti.x;
        tt.y = ti.y;
        tt.rf = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);
        tt.accum = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);
        tt.accum_weight = compute::vector<float>(T.W*T.W, ctx);
        size_t size;
        clfftGetTmpBufSize(T.ftplan, &size);
        tt.tmpbuf = compute::vector<char>(size, ctx);
    }
}

void prepare_image(image& image, const img_t<float>& img, things& T)
{
    auto ctx = T.queue.get_context();
    image.src = img;
    image.w = img.w;
    image.h = img.h;
    image.d = img.d;

    if (!image.allocated) {
        image.dev = img_to_device(img, T.queue);
        to_tiles_with_allocation(image, T);
        image.nt = image.tiles.size();

        for (int t = 0; t < image.nt; t++) {
            auto& tt = image.tiles[t];
            tt.h = compute::vector<float>(T.W*T.W*image.d, ctx);
            tt.f_sqr = compute::vector<float>(T.W*T.W*image.d, ctx);
            tt.ff = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);
            tt.fh = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);
            tt.cc = compute::vector<std::complex<float>>(T.W*T.W, ctx);
            tt.l2residuals = compute::vector<float>(T.W*T.W, ctx);
            tt.rff = compute::vector<std::complex<float>>(T.W*T.W*image.d, ctx);
            tt.w = compute::vector<float>(T.W*T.W, ctx);
            tt.wblur = compute::vector<float>(T.W*T.W, ctx);
            tt.magn = compute::vector<float>(T.W*T.W, ctx);
            tt.boxfiltered = compute::vector<float>(T.W*T.W, ctx);
            tt.boxfiltered2 = compute::vector<float>(T.W*T.W, ctx);
            size_t size;
            clfftGetTmpBufSize(T.ftplan, &size);
            tt.tmpbuf = compute::vector<char>(size, ctx);
            clfftGetTmpBufSize(T.ftplangray, &size);
            tt.tmpbufgray = compute::vector<char>(size, ctx);
        }
        image.allocated = true;
    } else {
        compute::copy(img.data.begin(), img.data.end(), image.dev.begin(), T.queue);
        to_tiles_without_allocation(image, T);
    }

    fulltohalf(image, T);
    fftfull(image, T);
    ffthalf(image, T);
    weight(image, T);
    boxfilter(image, T);
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 4) {
        return fprintf(stderr, "usage: %s <output_fmt> [file_of_inputs (stdin)]\n", argv[0]), 1;
    }

    char* output_fmt = argv[1];
    FILE* inputs = stdin;
    if (argc == 3) {
        inputs = fopen(argv[2], "r");
        if (!inputs) {
            return perror(argv[2]), 1;
        }
    }

    compute::device device = compute::system::default_device();
    std::cout << "device: " << device.name() << std::endl;

    compute::context ctx(device);

    int W = 256;

    things things;
    things.queue = compute::command_queue(ctx, device);
    things.W = W;
    things.O = W/3;
    things.p = 3;

    {
        img_t<float> _hann;
        ::hann(_hann, W/2, W/2);
        img_t<float> hann(W, W);
        hann.set_value(0);
        for (int y = 0; y < W/2; y++) {
            for (int x = 0; x < W/2; x++) {
                hann(W/4 + x, W/4 + y) = _hann(x, y);
            }
        }
        things.hann = img_to_device(hann, things.queue);
    }

    {
        float sigma = things.W / 50.f;
        std::vector<float> gaussian(21);
        float sum = 0.f;
        for (int x = 0; x < (int) gaussian.size(); x++) {
            gaussian[x] = 1.f/std::sqrt(2*M_PI*sigma*sigma)
                        * std::exp(- std::pow((float)(x-(int)gaussian.size()/2), 2.f) / (2*sigma*sigma));
            sum += gaussian[x];
        }
        for (unsigned x = 0; x < gaussian.size(); x++) {
            gaussian[x] /= sum;
        }
        things.gaussian = compute::vector<float>(gaussian.begin(), gaussian.end(), things.queue);
    }

    things.prog = compute::program::build_with_source(kernels_src, ctx,
                                                      "-D W=" + std::to_string(W)
                                                      + " -D hw=" + std::to_string(W/4));

    {
        clfftPlanHandle planHandle;
        clfftDim dim = CLFFT_2D;
        size_t clLengths[2] = {(size_t)W, (size_t)W};
        size_t strides[] = {(size_t)3, (size_t)W*3};
        int err;
        clfftSetupData fftSetup;
        err = clfftInitSetupData(&fftSetup);
        err = clfftSetup(&fftSetup);
        err = clfftCreateDefaultPlan(&things.ftplan, ctx, dim, clLengths);
        err = clfftSetPlanPrecision(things.ftplan, CLFFT_SINGLE);
        err = clfftSetLayout(things.ftplan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        err = clfftSetResultLocation(things.ftplan, CLFFT_INPLACE);
        err = clfftSetPlanInStride(things.ftplan, dim, strides);
        err = clfftSetPlanOutStride(things.ftplan, dim, strides);
        err = clfftSetPlanBatchSize(things.ftplan, 3);
        err = clfftSetPlanDistance(things.ftplan, 1, 1);
        cl_command_queue q = things.queue;
        err = clfftBakePlan(things.ftplan, 1, &q, NULL, NULL);
        assert(!err);
    }
    {
        clfftPlanHandle planHandle;
        clfftDim dim = CLFFT_2D;
        size_t clLengths[2] = {(size_t)W, (size_t)W};
        size_t strides[] = {(size_t)1, (size_t)W};
        int err;
        clfftSetupData fftSetup;
        err = clfftInitSetupData(&fftSetup);
        err = clfftSetup(&fftSetup);
        err = clfftCreateDefaultPlan(&things.ftplangray, ctx, dim, clLengths);
        err = clfftSetPlanPrecision(things.ftplangray, CLFFT_SINGLE);
        err = clfftSetLayout(things.ftplangray, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        err = clfftSetResultLocation(things.ftplangray, CLFFT_INPLACE);
        err = clfftSetPlanInStride(things.ftplangray, dim, strides);
        err = clfftSetPlanOutStride(things.ftplangray, dim, strides);
        err = clfftSetPlanBatchSize(things.ftplangray, 1);
        err = clfftSetPlanDistance(things.ftplangray, 1, 1);
        cl_command_queue q = things.queue;
        err = clfftBakePlan(things.ftplangray, 1, &q, NULL, NULL);
        assert(!err);
    }

    std::vector<struct image*> images;
    struct result result;

    int N = 3;
    int cur = 0;
    int i = 0;
    char file[2048];
    while (fgets(file, sizeof(file), inputs) && file[0]) {
        file[strlen(file) - 1] = 0;
        img_t<float> img = img_t<float>::load(file);
        float max = img.max();
        for (auto& v : img.data) v /= max;

        if (images.size() < N) {
            images.push_back(new struct image);
            images[images.size()-1]->allocated = false;
            cur = images.size() - 1;
        } else {
            cur = (cur + 1) % images.size();
        }

        prepare_image(*images[cur], img, things);

        if (images.size() == 1)
            initialize_result(result, *images[0], things);

        register_all(*images[cur], images, things);

        fuse_all(result, images, things);

        auto accumulated = img_from_device(result.accumulated, img.w, img.h, img.d, things.queue);
        for (auto& v : accumulated.data) v *= max;
        std::string output = string_format(output_fmt, i);
        accumulated.save(output);
        printf("%s\n", output.c_str());
        i++;
    }

    int err = clfftDestroyPlan(&things.ftplan);
    err = clfftDestroyPlan(&things.ftplangray);
    clfftTeardown();
    return 0;
}
