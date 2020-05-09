#include <iostream>
#include <vkc/vkc.hpp>

int main()
{
    vkc::init(800, 600, "Test");
    vkc::loopRun();
    return 0;
}
