// Entry point bridging Windows main (WinMain) and standard main signatures.

#include "Application.hpp"

#include <exception>
#include <iostream>

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
