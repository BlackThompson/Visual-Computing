#version 330 core

// GPU implementation of the pixelation filter. The shader groups pixels into
// square blocks in texture space and samples the texel located at the centre of
// each block. The block size is expressed in screen pixels which allows the GUI
// to match the CPU implementation.

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uFrame;
uniform vec2 uTextureSize;     // Width/height in pixels, used for block sizing.
uniform float uPixelBlockSize;
uniform mat3 uTexTransform;
uniform int uTransformEnabled;

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

void main()
{
    bool outside = false;
    vec2 uv = applyTransform(vTexCoord, outside);

    if (outside)
    {
        FragColor = vec4(0.0);
        return;
    }

    // Convert to absolute pixel space, quantise to the requested block size,
    // then convert back to normalised coordinates for the texture lookup.
    float blockSize = max(uPixelBlockSize, 1.0);
    vec2 pixelSpace = uv * uTextureSize;
    vec2 blockCoord = floor(pixelSpace / blockSize) * blockSize + blockSize / 2.0;
    vec2 sampleUv = blockCoord / uTextureSize;

    FragColor = texture(uFrame, sampleUv);
}
