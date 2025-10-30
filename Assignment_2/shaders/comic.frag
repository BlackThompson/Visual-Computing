#version 330 core

// Stylised "comic book" shader. The filter combines three operations:
//  1. Apply the optional affine transform to align with the CPU pipeline.
//  2. Quantise the colour palette into a handful of discrete bands.
//  3. Detect high-contrast edges with a Sobel kernel and multiply them back
//     into the colour image to emphasise outlines.

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uFrame;
uniform vec2 uTextureSize;
uniform mat3 uTexTransform;
uniform int uTransformEnabled;
uniform int uColorLevels;
uniform float uEdgeThreshold;

const vec3 kLumaWeights = vec3(0.299, 0.587, 0.114);

vec2 applyTransform(vec2 uv, out bool outside)
{
    outside = false;
    if (uTransformEnabled == 0)
    {
        return uv;
    }

    vec2 pixels = vec2(uv.x * uTextureSize.x,
                       (1.0 - uv.y) * uTextureSize.y);
    vec3 mapped = uTexTransform * vec3(pixels, 1.0);

    vec2 transformed = vec2(mapped.x / uTextureSize.x,
                            1.0 - (mapped.y / uTextureSize.y));

    if (transformed.x < 0.0 || transformed.x > 1.0 ||
        transformed.y < 0.0 || transformed.y > 1.0)
    {
        outside = true;
    }

    return transformed;
}

float sampleLuma(vec2 uv)
{
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    return dot(texture(uFrame, uv).rgb, kLumaWeights);
}

void main()
{
    bool outside = false;
    vec2 uv = applyTransform(vTexCoord, outside);

    if (outside)
    {
        FragColor = vec4(0.0);
        return;
    }

    vec3 colour = texture(uFrame, clamp(uv, vec2(0.0), vec2(1.0))).rgb;

    float levels = max(float(uColorLevels), 2.0);
    vec3 quantised = floor(colour * levels) / (levels - 1.0);

    vec2 texel = 1.0 / uTextureSize;
    // Sobel operator for edge detection in texture space.
    float tl = sampleLuma(uv + texel * vec2(-1.0,  1.0));
    float tc = sampleLuma(uv + texel * vec2( 0.0,  1.0));
    float tr = sampleLuma(uv + texel * vec2( 1.0,  1.0));
    float ml = sampleLuma(uv + texel * vec2(-1.0,  0.0));
    float mr = sampleLuma(uv + texel * vec2( 1.0,  0.0));
    float bl = sampleLuma(uv + texel * vec2(-1.0, -1.0));
    float bc = sampleLuma(uv + texel * vec2( 0.0, -1.0));
    float br = sampleLuma(uv + texel * vec2( 1.0, -1.0));

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    float gradient = sqrt(gx * gx + gy * gy);

    float edgeMask = smoothstep(uEdgeThreshold, uEdgeThreshold * 4.0, gradient);
    float ink = 1.0 - edgeMask;

    FragColor = vec4(quantised * ink, 1.0);
}
