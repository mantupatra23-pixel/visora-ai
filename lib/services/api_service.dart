import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://visora-ai-5nqs.onrender.com";

  // 🧠 Common GET
  static Future<dynamic> get(String endpoint) async {
    final url = Uri.parse("$baseUrl$endpoint");
    final res = await http.get(url);
    if (res.statusCode == 200) {
      try {
        return jsonDecode(res.body);
      } catch (_) {
        return res.body;
      }
    } else {
      throw Exception("GET $endpoint failed (${res.statusCode})");
    }
  }

  // 🧠 Create new Video Job
  static Future<dynamic> createVideoJob({
    required String title,
    required String script,
    Map<String, dynamic>? options,
  }) async {
    final url = Uri.parse("$baseUrl/api/jobs/create");
    final body = jsonEncode({
      "title": title,
      "script": script,
      "options": options ?? {},
    });
    final res = await http.post(url,
        headers: {"Content-Type": "application/json"}, body: body);
    if (res.statusCode == 200 || res.statusCode == 201) {
      return jsonDecode(res.body);
    } else {
      throw Exception("Video job creation failed (${res.statusCode})");
    }
  }

  // 🧠 Get Profile
  static Future<dynamic> getProfile() async {
    final url = Uri.parse("$baseUrl/api/users/me");
    final res = await http.get(url);
    if (res.statusCode == 200) return jsonDecode(res.body);
    throw Exception("Profile fetch failed (${res.statusCode})");
  }

  // 🧠 Logout / Clear token (for now, dummy)
  static Future<void> clearToken() async {
    return;
  }
}
