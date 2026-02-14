// MARK: - RunAnywhere OTA Fine-Tuning Client (Swift)

import Foundation

// MARK: - Configuration

/// Configuration for the OTA fine-tuning client.
///
/// Connects to your fine-tuning cloud server (laptop or remote)
/// to upload training data and receive LoRA adapters over-the-air.
///
/// ```swift
/// let config = OtaFineTuningConfig(
///     cloudEndpoint: "http://192.168.1.100:8000"
/// )
/// ```
public struct OtaFineTuningConfig {
    /// Cloud server endpoint URL (e.g., "http://192.168.1.100:8000")
    public let cloudEndpoint: String

    /// API key for authentication
    public var apiKey: String = "runanywhere-dev-key"

    /// Unique device identifier (auto-generated if nil)
    public var deviceId: String?

    /// Automatically upload interactions when threshold is reached
    public var autoUpload: Bool = true

    /// Minimum interactions before auto-upload triggers
    public var autoUploadThreshold: Int = 20

    /// Poll interval in seconds (0 = manual only)
    public var pollIntervalSeconds: Int = 60

    /// Automatically apply downloaded adapters
    public var autoApplyAdapter: Bool = true

    /// Max local adapter cache size in bytes (0 = unlimited)
    public var maxCacheSizeBytes: UInt64 = 0

    public init(
        cloudEndpoint: String,
        apiKey: String = "runanywhere-dev-key",
        deviceId: String? = nil,
        autoUpload: Bool = true,
        autoUploadThreshold: Int = 20,
        pollIntervalSeconds: Int = 60,
        autoApplyAdapter: Bool = true
    ) {
        self.cloudEndpoint = cloudEndpoint
        self.apiKey = apiKey
        self.deviceId = deviceId
        self.autoUpload = autoUpload
        self.autoUploadThreshold = autoUploadThreshold
        self.pollIntervalSeconds = pollIntervalSeconds
        self.autoApplyAdapter = autoApplyAdapter
    }
}

// MARK: - Models

/// A conversation context message.
public struct OtaContextMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// A training interaction captured from user chat.
public struct OtaTrainingInteraction: Codable, Sendable {
    public let id: String
    public let userPrompt: String
    public let modelResponse: String
    public var correctedResponse: String?
    public var rating: Int?
    public var isPositiveExample: Bool
    public var includeInTraining: Bool
    public var modelId: String?
    public var modelName: String?
    public var conversationContext: [OtaContextMessage]?
    public var category: String?
    public var tags: [String]
    public var timestamp: Int64
    public var frequencyCount: Int

    public init(
        id: String = UUID().uuidString,
        userPrompt: String,
        modelResponse: String,
        correctedResponse: String? = nil,
        rating: Int? = nil,
        isPositiveExample: Bool = false,
        includeInTraining: Bool = true,
        modelId: String? = nil,
        modelName: String? = nil,
        conversationContext: [OtaContextMessage]? = nil,
        category: String? = nil,
        tags: [String] = [],
        timestamp: Int64 = Int64(Date().timeIntervalSince1970 * 1000),
        frequencyCount: Int = 1
    ) {
        self.id = id
        self.userPrompt = userPrompt
        self.modelResponse = modelResponse
        self.correctedResponse = correctedResponse
        self.rating = rating
        self.isPositiveExample = isPositiveExample
        self.includeInTraining = includeInTraining
        self.modelId = modelId
        self.modelName = modelName
        self.conversationContext = conversationContext
        self.category = category
        self.tags = tags
        self.timestamp = timestamp
        self.frequencyCount = frequencyCount
    }

    enum CodingKeys: String, CodingKey {
        case id
        case userPrompt = "user_prompt"
        case modelResponse = "model_response"
        case correctedResponse = "corrected_response"
        case rating
        case isPositiveExample = "is_positive_example"
        case includeInTraining = "include_in_training"
        case modelId = "model_id"
        case modelName = "model_name"
        case conversationContext = "conversation_context"
        case category, tags, timestamp
        case frequencyCount = "frequency_count"
    }
}

/// Metadata for a trained LoRA adapter.
public struct OtaAdapterInfo: Codable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public var description: String?
    public let deviceId: String
    public let modelId: String
    public let baseModel: String
    public var format: String
    public var fileSizeBytes: Int64
    public var loraRank: Int
    public var loraAlpha: Int
    public var targetModules: [String]
    public var trainingSamples: Int
    public var finalLoss: Float?
    public var version: Int
    public var checksum: String?
    public var isPublished: Bool
    public var createdAt: String?

    enum CodingKeys: String, CodingKey {
        case id, name, description, format, version, checksum
        case deviceId = "device_id"
        case modelId = "model_id"
        case baseModel = "base_model"
        case fileSizeBytes = "file_size_bytes"
        case loraRank = "lora_rank"
        case loraAlpha = "lora_alpha"
        case targetModules = "target_modules"
        case trainingSamples = "training_samples"
        case finalLoss = "final_loss"
        case isPublished = "is_published"
        case createdAt = "created_at"
    }
}

/// Response from adapter polling.
public struct AdapterPollResponse: Codable, Sendable {
    public let hasNewAdapters: Bool
    public let adapters: [OtaAdapterInfo]
    public let pollIntervalSeconds: Int

    enum CodingKeys: String, CodingKey {
        case hasNewAdapters = "has_new_adapters"
        case adapters
        case pollIntervalSeconds = "poll_interval_seconds"
    }
}

/// Upload response.
public struct TrainingDataUploadResponse: Codable, Sendable {
    public let uploadId: String
    public let deviceId: String
    public let sampleCount: Int
    public let totalSamplesForDevice: Int
    public let message: String

    enum CodingKeys: String, CodingKey {
        case uploadId = "upload_id"
        case deviceId = "device_id"
        case sampleCount = "sample_count"
        case totalSamplesForDevice = "total_samples_for_device"
        case message
    }
}

/// Training job status response.
public struct TrainingJobStatusResponse: Codable, Sendable {
    public let jobId: String
    public let deviceId: String
    public let modelId: String
    public let baseModel: String
    public let status: String
    public var progress: Float
    public var currentEpoch: Int
    public var totalEpochs: Int
    public var currentLoss: Float?
    public var bestLoss: Float?
    public var errorMessage: String?
    public var adapterId: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case deviceId = "device_id"
        case modelId = "model_id"
        case baseModel = "base_model"
        case status, progress
        case currentEpoch = "current_epoch"
        case totalEpochs = "total_epochs"
        case currentLoss = "current_loss"
        case bestLoss = "best_loss"
        case errorMessage = "error_message"
        case adapterId = "adapter_id"
    }
}

// MARK: - Events

/// Events emitted by the OTA fine-tuning system.
public enum OtaFineTuningEvent: Sendable {
    case dataUploaded(uploadId: String, sampleCount: Int, totalSamples: Int)
    case adapterAvailable(adapter: OtaAdapterInfo)
    case downloadProgress(adapterId: String, progress: Float)
    case adapterDownloaded(adapter: OtaAdapterInfo, localPath: String)
    case adapterApplied(adapter: OtaAdapterInfo)
    case error(operation: String, message: String)
}

// MARK: - Client

/// OTA Fine-Tuning Client for iOS/macOS.
///
/// Handles uploading training data, polling for adapters,
/// downloading, and managing LoRA adapters locally.
public actor OtaFineTuningClient {
    private let config: OtaFineTuningConfig
    private let deviceId: String
    private let adapterStorageDir: URL
    private let session: URLSession

    private var localAdapters: [String: OtaAdapterInfo] = [:]
    private var activeAdapterId: String?
    private var pendingInteractions: [OtaTrainingInteraction] = []
    private var pollTask: Task<Void, Never>?
    private var lastPollTimestamp: String?

    /// Event stream — subscribe to receive OTA events.
    private let eventContinuation: AsyncStream<OtaFineTuningEvent>.Continuation
    public let events: AsyncStream<OtaFineTuningEvent>

    public init(
        config: OtaFineTuningConfig,
        adapterStorageDir: URL
    ) {
        self.config = config
        self.deviceId = config.deviceId ?? "ios-\(UUID().uuidString.prefix(8))"
        self.adapterStorageDir = adapterStorageDir
        self.session = URLSession(configuration: .default)

        // Create event stream
        var continuation: AsyncStream<OtaFineTuningEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.eventContinuation = continuation

        // Ensure storage directory exists
        try? FileManager.default.createDirectory(
            at: adapterStorageDir,
            withIntermediateDirectories: true
        )

        // Load persisted adapter registry
        loadAdapterRegistry()
    }

    deinit {
        pollTask?.cancel()
        eventContinuation.finish()
    }

    // MARK: - Training Data Upload

    /// Buffer an interaction for later upload.
    public func recordInteraction(_ interaction: OtaTrainingInteraction) async {
        pendingInteractions.append(interaction)

        if config.autoUpload && pendingInteractions.count >= config.autoUploadThreshold {
            let toUpload = pendingInteractions
            pendingInteractions.removeAll()
            _ = try? await uploadTrainingData(
                interactions: toUpload,
                modelId: interaction.modelId ?? ""
            )
        }
    }

    /// Upload training interactions to the cloud server.
    public func uploadTrainingData(
        interactions: [OtaTrainingInteraction],
        modelId: String,
        modelName: String? = nil
    ) async throws -> TrainingDataUploadResponse {
        struct UploadRequest: Encodable {
            let device_id: String
            let model_id: String
            let model_name: String?
            let interactions: [OtaTrainingInteraction]
        }

        let request = UploadRequest(
            device_id: deviceId,
            model_id: modelId,
            model_name: modelName,
            interactions: interactions
        )

        let response: TrainingDataUploadResponse = try await post(
            path: "/api/v1/training/upload",
            body: request
        )

        eventContinuation.yield(.dataUploaded(
            uploadId: response.uploadId,
            sampleCount: response.sampleCount,
            totalSamples: response.totalSamplesForDevice
        ))

        return response
    }

    /// Flush pending interactions.
    public func flushPendingInteractions(modelId: String) async throws -> TrainingDataUploadResponse? {
        guard !pendingInteractions.isEmpty else { return nil }
        let toUpload = pendingInteractions
        pendingInteractions.removeAll()
        return try await uploadTrainingData(interactions: toUpload, modelId: modelId)
    }

    // MARK: - Adapter Polling

    /// Poll the cloud server for new adapters.
    public func pollForAdapters(modelId: String? = nil) async throws -> AdapterPollResponse {
        var path = "/api/v1/adapters/poll?device_id=\(deviceId)"
        if let modelId { path += "&model_id=\(modelId)" }
        if let since = lastPollTimestamp { path += "&since=\(since)" }

        let response: AdapterPollResponse = try await get(path: path)

        if response.hasNewAdapters {
            for adapter in response.adapters {
                eventContinuation.yield(.adapterAvailable(adapter: adapter))

                if config.autoApplyAdapter {
                    _ = try? await downloadAdapter(adapterId: adapter.id)
                }
            }
        }

        lastPollTimestamp = ISO8601DateFormatter().string(from: Date())
        return response
    }

    /// Start automatic polling.
    public func startAutoPolling(modelId: String? = nil) {
        stopAutoPolling()
        pollTask = Task {
            while !Task.isCancelled {
                _ = try? await pollForAdapters(modelId: modelId)
                try? await Task.sleep(nanoseconds: UInt64(config.pollIntervalSeconds) * 1_000_000_000)
            }
        }
    }

    /// Stop automatic polling.
    public func stopAutoPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    // MARK: - Adapter Download

    /// Download an adapter from the cloud server.
    public func downloadAdapter(adapterId: String) async throws -> OtaAdapterInfo {
        // Get adapter metadata
        var adapterInfo: OtaAdapterInfo = try await get(path: "/api/v1/adapters/\(adapterId)")

        // Download file
        let adapterDir = adapterStorageDir.appendingPathComponent(adapterId)
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)

        let ext = adapterInfo.format == "gguf" ? "gguf" : "safetensors"
        let localFile = adapterDir.appendingPathComponent("adapter.\(ext)")

        try await downloadFile(
            path: "/api/v1/adapters/download/\(adapterId)",
            destination: localFile,
            onProgress: { [eventContinuation] progress in
                eventContinuation.yield(.downloadProgress(adapterId: adapterId, progress: progress))
            }
        )

        // Verify checksum
        if let expectedChecksum = adapterInfo.checksum {
            let localChecksum = try computeSHA256(url: localFile)
            guard localChecksum == expectedChecksum else {
                try? FileManager.default.removeItem(at: localFile)
                throw NSError(
                    domain: "OtaFineTuning",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Checksum mismatch"]
                )
            }
        }

        // Register locally
        localAdapters[adapterId] = adapterInfo
        saveAdapterRegistry()

        eventContinuation.yield(.adapterDownloaded(
            adapter: adapterInfo,
            localPath: localFile.path
        ))

        if config.autoApplyAdapter {
            applyAdapter(adapterId: adapterId)
        }

        return adapterInfo
    }

    // MARK: - Adapter Management

    /// Apply a downloaded adapter.
    public func applyAdapter(adapterId: String) {
        guard let adapter = localAdapters[adapterId] else { return }

        // Deactivate current
        if let currentId = activeAdapterId {
            // Note: localAdapters values are structs, need to update in dict
            activeAdapterId = nil
        }

        activeAdapterId = adapterId
        saveAdapterRegistry()

        eventContinuation.yield(.adapterApplied(adapter: adapter))
    }

    /// Remove active adapter.
    public func removeActiveAdapter() {
        activeAdapterId = nil
        saveAdapterRegistry()
    }

    /// Get all local adapters.
    public func getLocalAdapters() -> [OtaAdapterInfo] {
        Array(localAdapters.values)
    }

    /// Get active adapter.
    public func getActiveAdapter() -> OtaAdapterInfo? {
        guard let id = activeAdapterId else { return nil }
        return localAdapters[id]
    }

    /// Delete a local adapter.
    public func deleteLocalAdapter(adapterId: String) {
        if activeAdapterId == adapterId { removeActiveAdapter() }
        localAdapters.removeValue(forKey: adapterId)

        let adapterDir = adapterStorageDir.appendingPathComponent(adapterId)
        try? FileManager.default.removeItem(at: adapterDir)
        saveAdapterRegistry()
    }

    // MARK: - HTTP Helpers

    private func post<T: Encodable, R: Decodable>(path: String, body: T) async throws -> R {
        var request = URLRequest(url: URL(string: "\(config.cloudEndpoint)\(path)")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue(deviceId, forHTTPHeaderField: "X-Device-Id")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "OtaFineTuning", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: body])
        }

        let decoder = JSONDecoder()
        return try decoder.decode(R.self, from: data)
    }

    private func get<R: Decodable>(path: String) async throws -> R {
        var request = URLRequest(url: URL(string: "\(config.cloudEndpoint)\(path)")!)
        request.httpMethod = "GET"
        request.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue(deviceId, forHTTPHeaderField: "X-Device-Id")

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "OtaFineTuning", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: body])
        }

        return try JSONDecoder().decode(R.self, from: data)
    }

    private func downloadFile(
        path: String,
        destination: URL,
        onProgress: @Sendable @escaping (Float) -> Void
    ) async throws {
        var request = URLRequest(url: URL(string: "\(config.cloudEndpoint)\(path)")!)
        request.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key")

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NSError(domain: "OtaFineTuning", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Download failed"])
        }

        try data.write(to: destination)
        onProgress(1.0)
    }

    // MARK: - Persistence

    private var registryURL: URL {
        adapterStorageDir.appendingPathComponent("adapter_registry.json")
    }

    private func loadAdapterRegistry() {
        guard FileManager.default.fileExists(atPath: registryURL.path) else { return }
        guard let data = try? Data(contentsOf: registryURL),
              let adapters = try? JSONDecoder().decode([OtaAdapterInfo].self, from: data)
        else { return }

        for adapter in adapters {
            localAdapters[adapter.id] = adapter
        }
    }

    private func saveAdapterRegistry() {
        let adapters = Array(localAdapters.values)
        guard let data = try? JSONEncoder().encode(adapters) else { return }
        try? data.write(to: registryURL)
    }

    private func computeSHA256(url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        let digest = data.withUnsafeBytes { buffer -> [UInt8] in
            var hasher = _SHA256()
            hasher.update(data: buffer)
            return Array(hasher.finalize())
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

// Minimal SHA-256 using CryptoKit or CommonCrypto
import CryptoKit

private typealias _SHA256 = SHA256
