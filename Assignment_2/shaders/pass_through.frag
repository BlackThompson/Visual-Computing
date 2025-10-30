#version 330 core

// Basic fragment shader used for CPU processed frames. The shader simply
// samples the uploaded texture without applying any filter. The uniforms are
// kept consistent with the other shaders to simplify the renderer logic.

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uFrame;
uniform vec2 uTextureSize;         // Width/height in pixels, used for transform.
uniform mat3 uTexTransform;        // Row-equivalent affine matrix (OpenCV compatible).
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

    FragColor = texture(uFrame, uv);
}
