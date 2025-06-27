#include <unordered_map>
#include <string>
#include <iostream>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> 



inline std::shared_ptr<spdlog::logger> get_logger(std::string name)
{
    auto logger = spdlog::get(name);
    if (!logger) {
        logger = spdlog::stdout_color_mt(name);
        logger->set_pattern("%^[%l]%$\033[1;35m[%n]\033[0m %v");
    }

    return logger;
}

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
