#include <unordered_map>
#include <string>
#include <iostream>

template<typename T>
void print_unordered_map(const std::unordered_map<std::string, T> &map)
{
    std::cout << "{" << std::endl;
    for (const auto &pair : map)
    {
        std::cout << "\t" << pair.first << ": " << pair.second << std::endl;
    }
    std::cout << "}" << std::endl;
}
