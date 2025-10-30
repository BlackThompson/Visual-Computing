#version 330 core

// Vertex shader responsible for drawing the full-screen textured quad.
// The geometry is defined in clip-space already, so the vertex shader only
// needs to forward the texture coordinates downstream.

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
