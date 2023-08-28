#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Folder {
    std::string rgb_folder;
    std::string depth_folder;
    std::string seg_folder;
    int start_num;
    std::string object_class;
};


int count_images_in_folder(const std::string& folder_name, const std::regex& pattern) {
    int count = 0;
    for (const auto & entry : fs::directory_iterator(folder_name)) {
        if (std::regex_match(entry.path().filename().string(), pattern)) {
            count++;
        }
    }
    return count;
}

void process_folder(const std::string& src_rgb_folder, const std::string& src_depth_folder, const std::string& src_seg_folder, const std::string& dst_folder, const std::regex& rgb_pattern, const std::regex& depth_pattern, const std::regex& seg_pattern, int& start_num, const std::string& prefix, const std::string& extension) {
    std::vector<fs::directory_entry> rgb_entries, depth_entries, seg_entries;
    for (const auto & entry : fs::directory_iterator(src_rgb_folder)) {
        if (std::regex_match(entry.path().filename().string(), rgb_pattern)) {
            rgb_entries.push_back(entry);
        }
    }

    for (const auto & entry : fs::directory_iterator(src_depth_folder)) {
        if (std::regex_match(entry.path().filename().string(), depth_pattern)) {
            depth_entries.push_back(entry);
        }
    }

    for (const auto & entry : fs::directory_iterator(src_seg_folder)) {
        if (std::regex_match(entry.path().filename().string(), seg_pattern)) {
            seg_entries.push_back(entry);
        }
    }

    // Ensure that all folders have the same number of images
    assert(rgb_entries.size() == depth_entries.size() && depth_entries.size() == seg_entries.size());

    // Sort the entries based on the numerical part of their path
    auto sort_function = [](const fs::directory_entry& a, const fs::directory_entry& b) {
        std::string name_a = a.path().stem().string();
        std::string name_b = b.path().stem().string();
        name_a.erase(std::remove_if(name_a.begin(), name_a.end(), [](char c) { return !std::isdigit(c); }), name_a.end());
        name_b.erase(std::remove_if(name_b.begin(), name_b.end(), [](char c) { return !std::isdigit(c); }), name_b.end());
        int num_a = std::stoi(name_a);
        int num_b = std::stoi(name_b);
        return num_a < num_b;
    };

    std::sort(rgb_entries.begin(), rgb_entries.end(), sort_function);
    std::sort(depth_entries.begin(), depth_entries.end(), sort_function);
    std::sort(seg_entries.begin(), seg_entries.end(), sort_function);

    std::cout << "RGB entries: " << rgb_entries.size() << std::flush;
    std::cout << "Depth entries: " << depth_entries.size() << std::flush;
    std::cout << "Segmentation entries: " << seg_entries.size() << std::flush;


    // Process the sorted entries
    for (size_t i = 0; i < rgb_entries.size(); i++) {
        std::string new_filename_rgb = prefix + std::to_string(start_num) + ".bmp";
        std::string new_filename_depth = prefix + std::to_string(start_num) + ".png";
        std::string new_filename_seg = prefix + std::to_string(start_num) + ".bmp";
        fs::copy(rgb_entries[i].path(), dst_folder + "/rgb/" + new_filename_rgb);
        fs::copy(depth_entries[i].path(), dst_folder + "/depth/" + new_filename_depth);
        fs::copy(seg_entries[i].path(), dst_folder + "/seg/" + new_filename_seg);
        start_num++;
    }

}



int main() {
    std::regex rgb_pattern("\\d+\\.bmp");
    std::regex depth_pattern("d\\d+\\.png");
    std::regex seg_pattern("seg\\d+\\.bmp");
    std::vector<Folder> folders;
    std::string rgb_folder_path, depth_folder_path, seg_folder_path, object_class, continue_input;
    int next_start_num = 0;

    do {
        std::cout << "Please enter a path to the RGB folder: ";
        std::getline(std::cin, rgb_folder_path);

        std::cout << "Please enter a path to the Depth folder: ";
        std::getline(std::cin, depth_folder_path);

        std::cout << "Please enter a path to the Segmentation folder: ";
        std::getline(std::cin, seg_folder_path);

        std::cout << "Please enter the class for these folders: ";
        std::getline(std::cin, object_class);

        folders.push_back({rgb_folder_path, depth_folder_path, seg_folder_path, next_start_num, object_class});
        next_start_num += count_images_in_folder(rgb_folder_path, rgb_pattern);

        std::cout << "Do you want to add another folder? (yes/no) ";
        std::getline(std::cin, continue_input);
    } while (continue_input == "yes");


    std::string processed_folder;
    std::cout << "Please enter the processed folder path: ";
    std::cin >> processed_folder;
    fs::create_directories(processed_folder);

    for (auto &folder: folders) {
        process_folder(folder.rgb_folder, folder.depth_folder, folder.seg_folder, processed_folder, rgb_pattern,
                       depth_pattern, seg_pattern, folder.start_num, "", ".bmp");
    }

    nlohmann::json j;
    int last_end_frame = -1;
    for (const auto &folder: folders) {
        nlohmann::json j_folder;
        int num_frames_rgb = count_images_in_folder(folder.rgb_folder, rgb_pattern);
        int num_frames_depth = count_images_in_folder(folder.depth_folder, depth_pattern);
        int num_frames_seg = count_images_in_folder(folder.seg_folder, seg_pattern);
        j_folder["start_frame"] = last_end_frame + 1;
        last_end_frame = j_folder["start_frame"].get<int>() + num_frames_rgb -
                         1; // assuming all image types have the same number of frames
        j_folder["end_frame"] = last_end_frame;
        j_folder["rgb_num_frames"] = num_frames_rgb;
        j_folder["depth_num_frames"] = num_frames_depth;
        j_folder["seg_num_frames"] = num_frames_seg;
        j_folder["object_class"] = folder.object_class;
        j.push_back(j_folder);
    }

    std::ofstream output_file("classes.json");
    output_file << j.dump(4);
    return 0;
}

