import 'dart:convert';
import 'package:http/http.dart' as http;

const String apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'https://visora-ai-5nqs.onrender.com',
);

class ApiService {
  static Future<Map<String, dynamic>> _handleResponse(http.Response response) async {
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Error: ${response.statusCode}');
    }
  }

  // 🔹 User Login
  static Future<Map<String, dynamic>> login(String email, String password) async {
    final res = await http.post(
      Uri.parse('$apiBaseUrl/api/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    return _handleResponse(res);
  }

  // 🔹 User Registration
  static Future<Map<String, dynamic>> register(String name, String email, String password) async {
    final res = await http.post(
      Uri.parse('$apiBaseUrl/api/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name, 'email': email, 'password': password}),
    );
    return _handleResponse(res);
  }

  // 🔹 Get User Profile
  static Future<Map<String, dynamic>> getProfile(String token) async {
    final res = await http.get(
      Uri.parse('$apiBaseUrl/api/profile'),
      headers: {'Authorization': 'Bearer $token'},
    );
    return _handleResponse(res);
  }

  // 🔹 Create Video Job (AI Script → Video)
  static Future<Map<String, dynamic>> createVideoJob({
    required String title,
    required String script,
    String? language,
    String? quality,
  }) async {
    final res = await http.post(
      Uri.parse('$apiBaseUrl/api/create-job'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'title': title,
        'script': script,
        'language': language ?? 'en',
        'quality': quality ?? 'HD',
      }),
    );
    return _handleResponse(res);
  }

  // 🔹 Get Job Status
  static Future<Map<String, dynamic>> getJobStatus(String jobId) async {
    final res = await http.get(Uri.parse('$apiBaseUrl/api/job-status/$jobId'));
    return _handleResponse(res);
  }

  // 🔹 Fetch All Templates
  static Future<List<dynamic>> getTemplates() async {
    final res = await http.get(Uri.parse('$apiBaseUrl/api/templates'));
    return jsonDecode(res.body)['templates'] ?? [];
  }

  // 🔹 Fetch Dashboard Data
  static Future<Map<String, dynamic>> getDashboard() async {
    final res = await http.get(Uri.parse('$apiBaseUrl/api/dashboard'));
    return _handleResponse(res);
  }

  // 🔹 Update User Profile
  static Future<Map<String, dynamic>> updateProfile({
    required String token,
    required String name,
    required String email,
  }) async {
    final res = await http.put(
      Uri.parse('$apiBaseUrl/api/profile/update'),
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'name': name, 'email': email}),
    );
    return _handleResponse(res);
  }

  // 🔹 Clear Token / Logout
  static Future<void> clearToken(String token) async {
    await http.post(
      Uri.parse('$apiBaseUrl/api/logout'),
      headers: {'Authorization': 'Bearer $token'},
    );
  }
}
