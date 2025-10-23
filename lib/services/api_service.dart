import 'dart:async';

class ApiService {
  static Future<bool> loginWithEmail(String email, String password) async {
    await Future.delayed(const Duration(seconds: 1));
    return true;
  }

  static Future<String> getJobStatus(String jobId) async {
    await Future.delayed(const Duration(seconds: 1));
    return "Job Completed Successfully!";
  }
}
