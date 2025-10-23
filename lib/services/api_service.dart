import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://visora-ai-5nqs.onrender.com";

  // 🔹 Create Video Job
  static Future<String?> createVideoJob({
    required String title,
    required String script,
    String? voice,
    String? language,
    String? quality,
    Map<String, dynamic>? options,
  }) async {
    try {
      final body = {
        "title": title,
        "script": script,
        "voice": voice ?? "default",
        "language": language ?? "en",
        "quality": quality ?? "1080p",
      };
      if (options != null) body.addAll(options);

      final response = await http.post(
        Uri.parse("$baseUrl/api/video/create"),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['job_id']?.toString();
      } else {
        print("❌ API Error: ${response.statusCode} - ${response.body}");
        return null;
      }
    } catch (e) {
      print("⚠️ createVideoJob Exception: $e");
      return null;
    }
  }

  // 🔹 Generic GET (supports admin endpoints)
  static Future<dynamic> get(String endpoint, {Map<String, dynamic>? params}) async {
    try {
      final uri = Uri.parse("$baseUrl$endpoint").replace(queryParameters: params);
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print("❌ GET Error: ${response.statusCode}");
        return null;
      }
    } catch (e) {
      print("⚠️ GET Exception: $e");
      return null;
    }
  }

  // 🔹 Get Video Status
  static Future<String> getJobStatus(String jobId) async {
    try {
      final res = await http.get(Uri.parse("$baseUrl/api/video/status/$jobId"));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return data['status'] ?? 'unknown';
      }
      return 'error';
    } catch (e) {
      print("⚠️ getJobStatus Error: $e");
      return 'error';
    }
  }

  // 🔹 Get Video Result
  static Future<String?> getVideoResult(String jobId) async {
    try {
      final res = await http.get(Uri.parse("$baseUrl/api/video/result/$jobId"));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return data['video_url'];
      }
      return null;
    } catch (e) {
      print("⚠️ getVideoResult Error: $e");
      return null;
    }
  }

  // 🔹 Clear Token / Logout (Dummy)
  static Future<void> clearToken() async => Future.delayed(const Duration(milliseconds: 500));
}
