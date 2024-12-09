#include <fstream>
#include <vector>
#include <iostream>
#include <cctype>  // for std::isspace
#include <sstream>  // for std::istringstream
#include <algorithm>  // for std::find_if

inline bool WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data)
{
    FILE* fp = fopen(pFileName, "wb");
    if (fp)
    {
        if (data.size())
        {
            fwrite(&data[0], 1, data.size(), fp);
        }
        fclose(fp);
    }
    else
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has write access\n";
        return false;
    }
    return true;
}

inline bool ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image)
{
    FILE* fp = fopen(pFileName, "rb");
    if (!fp)
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has read access\n";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    const long fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (!fileLength)
    {
        std::cout << pFileName << " has zero length\n";
        fclose(fp);
        return false;
    }

    image.resize((size_t)fileLength);
    fread(&image[0], 1, image.size(), fp);
    fclose(fp);
    return true;
}



// 去除字符串中的所有空格并转换为整数
inline int stringToIntWithoutSpaces(const std::string& str) {
    std::string trimmedStr = str;

    // 去除起始空格
    trimmedStr.erase(trimmedStr.begin(), std::find_if(trimmedStr.begin(), trimmedStr.end(), [](unsigned char ch) { return !std::isspace(ch); }));

    // 去除结尾空格
    auto endPos = std::find_if(trimmedStr.rbegin(), trimmedStr.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
    trimmedStr.erase(endPos, trimmedStr.end());

    // 使用istringstream转换为int
    std::istringstream iss(trimmedStr);
    int result;
    if (!(iss >> result)) {
        // 如果转换失败，抛出异常
        throw std::invalid_argument("Cannot convert string to int: " + trimmedStr);
    }

    return result;
}