#include "helper_f.h"

int frame_count = 0;
int save_frame_feq = 5000;
bool save_rframe_lock = false;
bool save_eframe_lock = false;

void add_frame_count(){
    frame_count++;
}

int get_frame_count(){
    return frame_count;
}

int get_save_frame_feq(){
    return save_frame_feq;
}

bool get_eframe_lock(){
    return save_eframe_lock;
}

std::string filename_s = "C:\\Users\\Arnold\\Documents\\NSDI_RLVR\\build\\alvr_streamer_windows\\";

std::string get_path_head(){
    return filename_s;
}

void SaveTextureAsBytes(ID3D11DeviceContext* context, ID3D11Texture2D* texture, bool FFRed)
{
    if(save_rframe_lock){
        ID3D11Device* device;
        texture->GetDevice(&device);
        // Get texture description
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);

        // Create staging texture
        D3D11_TEXTURE2D_DESC stagingDesc = desc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* stagingTexture;
        device->CreateTexture2D(&stagingDesc, nullptr, &stagingTexture);

        // Copy texture to staging texture
        context->CopyResource(stagingTexture, texture);

        // Map staging texture to CPU memory
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        context->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);

        // Write texture to byte file
        std::string name = "rframe_";
        name += std::to_string(get_frame_count());
        name += ".bytes";
        const char* filename = (filename_s+name).c_str();
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        file.write((char*)mappedResource.pData, mappedResource.DepthPitch);
        //getting the entropy too
        std::ofstream file(filename_s+"entropy.csv", std::ios_base::app);
        const unsigned char* imageData = reinterpret_cast<const unsigned char*>(mappedResource.pData);
        const int numPixels = mappedResource.RowPitch / sizeof(unsigned char);
        std::vector<int> histogram(256, 0);
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char pixelValue = imageData[i];
            histogram[pixelValue]++;
        }

        double entropy = 0.0;
        const double numPixelsInv = 1.0 / numPixels;
        for (int count : histogram)
        {
            if (count > 0)
            {
                double probability = count * numPixelsInv;
                entropy -= probability * std::log2(probability);
            }
        }


        add_frame_count();
        // Unmap staging texture
        context->Unmap(stagingTexture, 0);

        // Release resources
        stagingTexture->Release();
    }
}