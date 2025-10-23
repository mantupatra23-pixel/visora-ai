import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'https://visora-ai-5nqs.onrender.com',
  );

  // 🔹 Create Video Job
  static Future<String?> createVideoJob({
    required String title,
    required String script,
    String? voice,
    String? language,
    String? quality,
  }) async {
    try {
      final response = await http.post(
        Uri.parse("$baseUrl/api/video/create"),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "title": title,
          "script": script,
          "voice": voice ?? "default",
          "language": language ?? "en",
          "quality": quality ?? "1080p",
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['job_id'];
      } else {
        print("❌ API Error: ${response.body}");
        return null;
      }
    } catch (e) {
      print("⚠️ Exception in createVideoJob: $e");
      return null;
    }
  }

  // 🔹 Get Video Job Status
  static Future<String> getJobStatus(String jobId) async {
    try {
      final response = await http.get(Uri.parse("$baseUrl/api/video/status/$jobId"));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['status'] ?? "unknown";
      } else {
        return "error";
      }
    } catch (e) {
      print("⚠️ getJobStatus error: $e");
      return "error";
    }
  }

  // 🔹 Get Final Video URL / Result
  static Future<String?> getVideoResult(String jobId) async {
    try {
      final response = await http.get(Uri.parse("$baseUrl/api/video/result/$jobId"));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['video_url'];
      }
      return null;
    } catch (e) {
      print("⚠️ getVideoResult error: $e");
      return null;
    }
  }
}
