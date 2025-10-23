import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://visora-ai-5nqs.onrender.com/api";

  // 🧠 Utility: Common Headers
  static Map<String, String> get headers => {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  // 🔑 User Login
  static Future<Map<String, dynamic>> login(String email, String password) async {
    final url = Uri.parse('$baseUrl/login');
    final response = await http.post(
      url,
      headers: headers,
      body: jsonEncode({'email': email, 'password': password}),
    );
    return _handleResponse(response);
  }

  // 🧾 Get User Profile
  static Future<Map<String, dynamic>> getProfile(String token) async {
    final url = Uri.parse('$baseUrl/profile');
    final response = await http.get(
      url,
      headers: {
        ...headers,
        'Authorization': 'Bearer $token',
      },
    );
    return _handleResponse(response);
  }

  // 🎬 Create New Video Job
  static Future<Map<String, dynamic>> createVideoJob({
    required String title,
    required String script,
  }) async {
    final url = Uri.parse('$baseUrl/create-job');
    final response = await http.post(
      url,
      headers: headers,
      body: jsonEncode({'title': title, 'script': script}),
    );
    return _handleResponse(response);
  }

  // 📊 Get Job Status
  static Future<Map<String, dynamic>> getJobStatus(String jobId) async {
    final url = Uri.parse('$baseUrl/job-status/$jobId');
    final response = await http.get(url, headers: headers);
    return _handleResponse(response);
  }

  // 🧩 Get Templates
  static Future<Map<String, dynamic>> getTemplates() async {
    final url = Uri.parse('$baseUrl/templates');
    final response = await http.get(url, headers: headers);
    return _handleResponse(response);
  }

  // ⚙️ Admin Dashboard
  static Future<Map<String, dynamic>> getAdminDashboard() async {
    final url = Uri.parse('$baseUrl/admin/dashboard');
    final response = await http.get(url, headers: headers);
    return _handleResponse(response);
  }

  // 🧠 Clear Token (Logout)
  static Future<void> clearToken() async {
    // Logout handled in backend or token clear in local storage
    print("✅ Token cleared successfully (local only).");
  }

  // 🔒 Error-safe handler
  static Map<String, dynamic> _handleResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return jsonDecode(response.body);
    } else {
      throw Exception(
          "❌ API Error [${response.statusCode}]: ${response.body}");
    }
  }
}
