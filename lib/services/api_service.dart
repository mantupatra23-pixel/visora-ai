import 'dart:async';

class ApiService {
  // Simulated delay for API calls
  static Future<void> delay() async =>
      await Future.delayed(const Duration(milliseconds: 500));

  // Create new video job
  static Future<String> createVideoJob({
    required String title,
    required String script,
  }) async {
    await delay();
    return "JOB_${DateTime.now().millisecondsSinceEpoch}";
  }

  // Get job status
  static Future<Map<String, dynamic>> getJobStatus(String jobId) async {
    await delay();
    return {"status": "completed", "jobId": jobId};
  }

  // Get templates
  static Future<List<Map<String, String>>> get(String endpoint) async {
    await delay();
    return [
      {"id": "1", "title": "AI Motivational Template"},
      {"id": "2", "title": "YouTube Shorts Auto Style"},
    ];
  }

  // Get Profile
  static Future<Map<String, String>> getProfile() async {
    await delay();
    return {"name": "Aimantuvya", "role": "Creator", "tier": "Premium"};
  }

  // Clear Token (Logout Simulation)
  static Future<void> clearToken() async {
    await delay();
  }

  // Edit or re-create video
  static Future<String> editVideoJob(String id, String request) async {
    await delay();
    return "Edited_$id";
  }

  // Generic endpoint support
  static Future<Map<String, dynamic>> getApi(String endpoint) async {
    await delay();
    return {"endpoint": endpoint, "status": "ok"};
  }
}
