// Minimal OpenGL shader compilation helper with detailed logging.
// Handles both vertex and fragment shader stages required by the project.

#pragma once

#include <glad/glad.h>

#include <string>
#include <stdexcept>
#include <vector>

class ShaderProgram
{
public:
    ShaderProgram() = default;

    ShaderProgram(const std::string& vertexSource,
                  const std::string& fragmentSource)
    {
        create(vertexSource, fragmentSource);
    }

    ~ShaderProgram()
    {
        if (program_ != 0)
        {
            glDeleteProgram(program_);
        }
    }

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&& other) noexcept
        : program_(other.program_)
    {
        other.program_ = 0;
    }

    ShaderProgram& operator=(ShaderProgram&& other) noexcept
    {
        if (this != &other)
        {
            if (program_ != 0)
            {
                glDeleteProgram(program_);
            }
            program_ = other.program_;
            other.program_ = 0;
        }
        return *this;
    }

    void create(const std::string& vertexSource,
                const std::string& fragmentSource)
    {
        GLuint vert = compile(GL_VERTEX_SHADER, vertexSource);
        GLuint frag = compile(GL_FRAGMENT_SHADER, fragmentSource);

        program_ = glCreateProgram();
        glAttachShader(program_, vert);
        glAttachShader(program_, frag);
        glLinkProgram(program_);

        GLint status = 0;
        glGetProgramiv(program_, GL_LINK_STATUS, &status);
        if (status != GL_TRUE)
        {
            std::string info = getProgramInfoLog(program_);
            glDeleteShader(vert);
            glDeleteShader(frag);
            glDeleteProgram(program_);
            program_ = 0;
            throw std::runtime_error("Failed to link shader program: " + info);
        }

        glDetachShader(program_, vert);
        glDetachShader(program_, frag);
        glDeleteShader(vert);
        glDeleteShader(frag);
    }

    void use() const
    {
        glUseProgram(program_);
    }

    [[nodiscard]] GLuint id() const
    {
        return program_;
    }

private:
    static GLuint compile(GLenum type, const std::string& source)
    {
        GLuint shader = glCreateShader(type);
        const char* src = source.c_str();
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        GLint status = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            std::string info = getShaderInfoLog(shader);
            glDeleteShader(shader);
            throw std::runtime_error("Failed to compile shader: " + info);
        }

        return shader;
    }

    static std::string getShaderInfoLog(GLuint shader)
    {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(length), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        return log;
    }

    static std::string getProgramInfoLog(GLuint program)
    {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(length), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        return log;
    }

    GLuint program_ = 0;
};
