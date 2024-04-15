#include "helper_f.h"

int frame_count = 0;
int save_frame_feq = 5000;
bool initialized_CS = true;
bool ReInitialize_CS  = false;
std::ofstream entropyFile;

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
        std::ofstream file(filename_s+"entropy.csv", std::ios_base::app);
        auto start = std::chrono::high_resolution_clock::now();
        ID3D11Device* device;
        texture->GetDevice(&device);
        // Get texture description
        D3D11_TEXTURE2D_DESC inputDesc;
        texture->GetDesc(&inputDesc);
        file << "test1" << std::endl;

        // Load the compute shader binary from file
        std::ifstream shaderFile("C:\\Users\\User\\Documents\\NSDI_RLVR\\alvr\\server\\cpp\\analyze_use\\test.cso", std::ios::binary);
        std::vector<char> shaderData((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
        file << "test2" << std::endl;
        file << shaderData.size() << std::endl;

        // Create the compute shader object
        ID3D11ComputeShader* computeShader = nullptr;
        HRESULT hr = device->CreateComputeShader(shaderData.data(), shaderData.size(), nullptr, &computeShader);
        file << "test3" << std::endl;
        if FAILED(hr)
            file << "error3" << std::endl;
        ID3D11ShaderResourceView* inputTextureSRV;
        ID3D11Texture2D* inputTexture = nullptr;
        hr = device->CreateTexture2D(&inputDesc, nullptr, &inputTexture);
        context->CopyResource(inputTexture, texture);
        
        D3D11_SHADER_RESOURCE_VIEW_DESC inputSRVDesc = {};
        inputSRVDesc.Format = inputDesc.Format;
        inputSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        inputSRVDesc.Texture2D.MipLevels = inputDesc.MipLevels;
        inputSRVDesc.Texture2D.MostDetailedMip = 0;
        hr = device->CreateShaderResourceView(inputTexture, &inputSRVDesc, &inputTextureSRV);
        file << "test4  " << inputDesc.Width << ", " << inputDesc.Height << std::endl;
        if FAILED(hr)
            file << "error4" << std::endl;

        D3D11_BUFFER_DESC bufferGPUDesc;
        bufferGPUDesc.ByteWidth = sizeof(UINT)*256;
        bufferGPUDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferGPUDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        bufferGPUDesc.CPUAccessFlags = 0;
        bufferGPUDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bufferGPUDesc.StructureByteStride = sizeof(UINT);
        ID3D11Buffer* histogramBufferGPU;
        hr = device->CreateBuffer(&bufferGPUDesc, nullptr, &histogramBufferGPU);


        if (FAILED(hr))
        {
            file << "createTexture" << std::endl;
            // Handle error
            DWORD error = HRESULT_CODE(hr);
            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);
            if (size > 0)
            {
                file << "Failed to map histogram buffer for reading: " << messageBuffer << std::endl;
                LocalFree(messageBuffer);
            }
            else
            {
                file << "Failed to map histogram buffer for reading: " << error << std::endl;
            }
            return;
        }

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = 256; // assuming 256 bins
        uavDesc.Buffer.Flags = 0;
        ID3D11UnorderedAccessView* histogramUAV;
        hr = device->CreateUnorderedAccessView(histogramBufferGPU, &uavDesc, &histogramUAV);
        if (FAILED(hr))
        {
            file << "createUnorder" << std::endl;
            // Handle error
            DWORD error = HRESULT_CODE(hr);
            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);
            if (size > 0)
            {
                file << "Failed to map histogram buffer for reading: " << messageBuffer << std::endl;
                LocalFree(messageBuffer);
            }
            else
            {
                file << "Failed to map histogram buffer for reading: " << error << std::endl;
            }
            return;
        }

        context->CSSetShaderResources(0, 1, &inputTextureSRV);
        context->CSSetUnorderedAccessViews(0, 1, &histogramUAV, nullptr);

        context->CSSetShader(computeShader, nullptr, 0);
        file << "test8" << std::endl;

        UINT dispatchWidth = inputDesc.Width / 16;
        UINT dispatchHeight = inputDesc.Height / 16;
        context->Dispatch(dispatchWidth, dispatchHeight, 1);
        // context->Flush();
        file << "test10" << std::endl;

        // Create a CPU-accessible buffer for the histogram
        D3D11_BUFFER_DESC bufferDesc;
        bufferDesc.ByteWidth = sizeof(UINT)*256; // assuming 256 bins
        bufferDesc.Usage = D3D11_USAGE_STAGING;
        bufferDesc.BindFlags = 0;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        bufferDesc.MiscFlags = 0;
        ID3D11Buffer* histogramBuffer;
        hr = device->CreateBuffer(&bufferDesc, nullptr, &histogramBuffer);

        // Copy the histogram data from GPU to CPU
        context->CopyResource(histogramBuffer, histogramBufferGPU);

        // Map staging texture for reading
        D3D11_MAPPED_SUBRESOURCE mappedOutputTexture;
        hr = context->Map(histogramBuffer, 0, D3D11_MAP_READ, 0, &mappedOutputTexture);
        file << "test11" << std::endl;
        if(FAILED(hr)){
            file << "failed" << std::endl;
            DWORD error = HRESULT_CODE(hr);
            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);
            if (size > 0)
            {
                file << "Failed to map histogram buffer for reading: " << messageBuffer << std::endl;
                LocalFree(messageBuffer);
            }
            else
            {
                file << "Failed to map histogram buffer for reading: " << error << std::endl;
            }
            return;
        }
        file << mappedOutputTexture.DepthPitch << std::endl;
        UINT* histogramData = reinterpret_cast<UINT*>(mappedOutputTexture.pData);
        double entropy = 0.0;
        const int numPixels = inputDesc.Width*inputDesc.Height;
        int counter = 0;
        const double numPixelsInv = 1.0 / numPixels;
        for (int i=0; i<256; i++)
        {
            int count = (int)histogramData[i];
            counter += count;
            if (count > 0)
            {
                double probability = float(count) * numPixelsInv;
                entropy -= probability * std::log2(probability);
            }
        }
        file << entropy << ", " << counter << std::endl;


        context->Unmap(histogramBuffer, 0);
        file.close();
    }
}

ID3D11Device* p_Device = nullptr;
UINT dispatchWidth;
UINT dispatchHeight;
D3D11_TEXTURE2D_DESC inputDesc;
ID3D11ComputeShader* computeShaderHistogram = nullptr;
ID3D11ComputeShader* computeShaderEntropy = nullptr;
ID3D11Texture2D* inputTexture = nullptr;
ID3D11ShaderResourceView* inputTextureSRV;
D3D11_SHADER_RESOURCE_VIEW_DESC inputSRVDesc = {};
D3D11_BUFFER_DESC bufferGPUDesc;
ID3D11Buffer* histogramBufferGPU;
D3D11_BUFFER_DESC entropyGPUDesc;
ID3D11Buffer* entropyGPU;
D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
ID3D11UnorderedAccessView* histogramUAV;
D3D11_UNORDERED_ACCESS_VIEW_DESC uavEntropyDesc;
ID3D11UnorderedAccessView* EntropyUAV;
D3D11_BUFFER_DESC bufferDesc;
ID3D11Buffer* histogramBuffer;
D3D11_BUFFER_DESC EntropyDesc;
ID3D11Buffer* EntropyBuffer;
HRESULT hr;
UINT Zero[256] = {0};
UINT thread_use = 32;

void CalculateEntropy(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, uint64_t m_targetTimestampNs){
    auto start = std::chrono::high_resolution_clock::now();
    if(p_Device==nullptr){
        initialized_CS = true;
        p_Device = device;
    }
    if(device!=p_Device){
        ReInitialize_CS = true;
        initialized_CS = true;
        p_Device = device;
    }
    if(initialized_CS){
    //initialize
        initialized_CS = false;
        entropyFile.open(filename_s+"entropy.csv", std::ios_base::app);
        // entropyFile << "openFile" << std::endl;
        entropyFile << "target_ts(nanos),complexity,timeA,timeB,timeC,timeD,timeE,timeF,timeG,timeH,timeI,timeJ,timeK,timeL"

        if(ReInitialize_CS){
            entropyFile << "Reinitialized" << std::endl;
        }

        // Get input texture description
        texture->GetDesc(&inputDesc);

        // Load the compute shader binary from file for Histogram
        std::ifstream shaderFileHistogram(filename_s+"..\\..\\alvr\\server\\cpp\\analyze_use\\Histogram.cso", std::ios::binary);
        std::vector<char> shaderDataHistogram((std::istreambuf_iterator<char>(shaderFileHistogram)), std::istreambuf_iterator<char>());

        // Create the compute Shader object for Histogram
        hr = device->CreateComputeShader(shaderDataHistogram.data(), shaderDataHistogram.size(), nullptr, &computeShaderHistogram);
        if(FAILED(hr)){
            entropyFile << "Failed to create Shader" << std::endl;
        }

        // Load the compute shader binary from file for Entropy
        std::ifstream shaderFileEntropy(filename_s+"..\\..\\alvr\\server\\cpp\\analyze_use\\Entropy.cso", std::ios::binary);
        std::vector<char> shaderDataEntropy((std::istreambuf_iterator<char>(shaderFileEntropy)), std::istreambuf_iterator<char>());

        // Create the compute Shader object for Histogram
        hr = device->CreateComputeShader(shaderDataEntropy.data(), shaderDataEntropy.size(), nullptr, &computeShaderEntropy);
        if(FAILED(hr)){
            entropyFile << "Failed to create Shader" << std::endl;
        }

        //Create input texture for copy in
        hr = device->CreateTexture2D(&inputDesc, nullptr, &inputTexture);
        if(FAILED(hr)){
            entropyFile << "Failed to create Input texture" << std::endl;
        }

        // Create input texture SRV
        inputSRVDesc.Format = inputDesc.Format;
        inputSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        inputSRVDesc.Texture2D.MipLevels = inputDesc.MipLevels;
        inputSRVDesc.Texture2D.MostDetailedMip = 0;
        hr = device->CreateShaderResourceView(inputTexture, &inputSRVDesc, &inputTextureSRV);
        if(FAILED(hr)){
            entropyFile << "Failed to create SRV" << std::endl;
        }
        dispatchWidth = inputDesc.Width / thread_use;
        dispatchHeight = inputDesc.Height / thread_use;

        // Create buffer for GPU
        bufferGPUDesc.ByteWidth = sizeof(UINT)*256;
        bufferGPUDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferGPUDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        bufferGPUDesc.CPUAccessFlags = 0;
        bufferGPUDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bufferGPUDesc.StructureByteStride = sizeof(UINT);
        hr = device->CreateBuffer(&bufferGPUDesc, nullptr, &histogramBufferGPU);
        if(FAILED(hr)){
            entropyFile << "Failed to Create Buffer" << std::endl;
        }

        // Create UAV for buffer
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = 256; // assuming 256 bins
        uavDesc.Buffer.Flags = 0;
        hr = device->CreateUnorderedAccessView(histogramBufferGPU, &uavDesc, &histogramUAV);
        if(FAILED(hr)){
            entropyFile << "Failed to create UAV" << std::endl;
        }

        // Create buffer for Entropy
        entropyGPUDesc.ByteWidth = sizeof(float);
        entropyGPUDesc.Usage = D3D11_USAGE_DEFAULT;
        entropyGPUDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        entropyGPUDesc.CPUAccessFlags = 0;
        entropyGPUDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        entropyGPUDesc.StructureByteStride = sizeof(float);
        hr = device->CreateBuffer(&entropyGPUDesc, nullptr, &entropyGPU);
        if(FAILED(hr)){
            entropyFile << "Failed to Create entropy" << std::endl;
        }

        // Create UAV for entropy
        uavEntropyDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavEntropyDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavEntropyDesc.Buffer.FirstElement = 0;
        uavEntropyDesc.Buffer.NumElements = 1; // assuming 256 bins
        uavEntropyDesc.Buffer.Flags = 0;
        hr = device->CreateUnorderedAccessView(entropyGPU, &uavEntropyDesc, &EntropyUAV);
        if(FAILED(hr)){
            entropyFile << "Failed to create UAV" << std::endl;
        }

        // Create Mapping staging texture 
        // bufferDesc.ByteWidth = sizeof(UINT)*256; // assuming 256 bins
        // bufferDesc.Usage = D3D11_USAGE_STAGING;
        // bufferDesc.BindFlags = 0;
        // bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        // bufferDesc.MiscFlags = 0;
        // hr = device->CreateBuffer(&bufferDesc, nullptr, &histogramBuffer);
        // if(FAILED(hr)){
        //     entropyFile << "Failed to create staging buffer" << std::endl;
        // }

        // Create Mapping staging texture 
        EntropyDesc.ByteWidth = sizeof(float); // assuming 256 bins
        EntropyDesc.Usage = D3D11_USAGE_STAGING;
        EntropyDesc.BindFlags = 0;
        EntropyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        EntropyDesc.MiscFlags = 0;
        hr = device->CreateBuffer(&EntropyDesc, nullptr, &EntropyBuffer);
        if(FAILED(hr)){
            entropyFile << "Failed to create staging buffer" << std::endl;
        }

    }
    auto t1 = std::chrono::high_resolution_clock::now();
    context->CopyResource(inputTexture, texture);
    auto t2 = std::chrono::high_resolution_clock::now();
    context->CSSetShaderResources(0, 1, &inputTextureSRV);
    auto t3 = std::chrono::high_resolution_clock::now();
    context->ClearUnorderedAccessViewUint(histogramUAV, Zero);
    auto t4 = std::chrono::high_resolution_clock::now();
    context->CSSetUnorderedAccessViews(0, 1, &histogramUAV, nullptr);
    context->CSSetUnorderedAccessViews(1, 1, &EntropyUAV, nullptr);
    auto t5 = std::chrono::high_resolution_clock::now();

    context->CSSetShader(computeShaderHistogram, nullptr, 0);
    auto t6 = std::chrono::high_resolution_clock::now();
    context->Dispatch(dispatchWidth, dispatchHeight, 1);
    auto t7 = std::chrono::high_resolution_clock::now();

    context->CSSetShader(computeShaderEntropy, nullptr, 0);
    context->Dispatch(1,1,1);

    // context->CopyResource(histogramBuffer, histogramBufferGPU);
    context->CopyResource(EntropyBuffer, entropyGPU);
    auto t8 = std::chrono::high_resolution_clock::now();


    D3D11_MAPPED_SUBRESOURCE mappedOutputTexture;
    hr = context->Map(EntropyBuffer, 0, D3D11_MAP_READ, 0, &mappedOutputTexture);
    auto t9 = std::chrono::high_resolution_clock::now();
    if(FAILED(hr)){
        entropyFile << "failed mapping" << std::endl;
        DWORD error = HRESULT_CODE(hr);
        LPSTR messageBuffer = nullptr;
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);
        if (size > 0)
        {
            entropyFile << "Failed to map histogram buffer for reading: " << messageBuffer << std::endl;
            LocalFree(messageBuffer);
        }
        else
        {
            entropyFile << "Failed to map histogram buffer for reading: " << error << std::endl;
        }
        return; 
    }

    // UINT* histogramData = reinterpret_cast<UINT*>(mappedOutputTexture.pData);
    float* EntropyData = reinterpret_cast<float*>(mappedOutputTexture.pData);
    auto t10 = std::chrono::high_resolution_clock::now();
    // double entropy = 0.0;
    // const int numPixels = inputDesc.Width*inputDesc.Height;
    // int counter = 0;
    // const double numPixelsInv = 1.0 / numPixels;
    // for (int i=0; i<256; i++)
    // {
    //     int count = (int)histogramData[i];
    //     counter += count;
    //     if (count > 0)
    //     {
    //         double probability = float(count) * numPixelsInv;
    //         entropy -= probability * std::log2(probability);
    //     }
    // }
    auto t11 = std::chrono::high_resolution_clock::now();
    entropyFile << EntropyData[0] << ", " 
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t1-start).count() 
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() 
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t4-t3).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t5-t4).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t6-t5).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t7-t6).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t8-t7).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t9-t8).count()
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t10-t9).count() 
    << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(t11-t10).count() << std::endl;
    context->Unmap(EntropyBuffer, 0);
}

void CloseFile(){ //Cant work
    entropyFile << "testing" << std::endl;
    entropyFile.close();
}
