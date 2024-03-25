#include "helper_f.h"

int frame_count = 0;
int save_frame_feq = 5000;


void add_frame_count(){
    frame_count++;
}

int get_frame_count(){
    return frame_count;
}

int get_save_frame_feq(){
    return save_frame_feq;
}

void SaveTextureAsBytes(ID3D11DeviceContext* context, ID3D11Texture2D* texture, bool FFRed, uint64_t m_targetTimestampNs)
{
    if(get_rframe_lock()){
        auto start = std::chrono::high_resolution_clock::now();
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
        auto t1 = std::chrono::high_resolution_clock::now();
        // Write texture to byte file
        std::string name = "rframe_";
        name += std::to_string(get_frame_count());
        name += ".bytes";
        const char* filename = (filename_s+name).c_str();
        
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        file.write((char*)mappedResource.pData, mappedResource.DepthPitch);

        auto t2 = std::chrono::high_resolution_clock::now();
        //getting the entropy too
        std::ofstream file2(filename_s+"entropy.csv", std::ios_base::app);
        const unsigned char* imageData = reinterpret_cast<const unsigned char*>(mappedResource.pData);
        std::vector<int> histogram(256, 0);
        const int numPixels = mappedResource.DepthPitch / sizeof(unsigned char) / 4; // Assuming 4 bytes per pixel (RGBA)

        // Convert RGBA image to grayscale and calculate the histogram of pixel values
        for (int i = 0; i < numPixels; i++)
        {
            unsigned char red = imageData[i * 4];
            unsigned char green = imageData[i * 4 + 1];
            unsigned char blue = imageData[i * 4 + 2];
            
            // Calculate grayscale value using the luminosity method
            unsigned char grayscaleValue = static_cast<unsigned char>(0.299 * red + 0.587 * green + 0.114 * blue);
            
            histogram[grayscaleValue]++;
        }

        double entropy = 0.0;
        const double numPixelsInv = 1.0 / numPixels;
        for (int count : histogram)
        {
            if (count > 0)
            {
                double probability = float(count) * numPixelsInv;
                entropy -= probability * std::log2(probability);
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        file2 << m_targetTimestampNs << "," << frame_count << "," << entropy << "," << t1-start << "," << t2-t1 << "," << t3-t2 << std::endl;
        file2.close();

        add_frame_count();
        // Unmap staging texture
        context->Unmap(stagingTexture, 0);

        // Release resources
        stagingTexture->Release();
    }
}