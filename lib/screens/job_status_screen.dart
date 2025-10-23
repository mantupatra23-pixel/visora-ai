import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'dart:async';

class JobStatusScreen extends StatefulWidget {
  final String jobId;
  const JobStatusScreen({super.key, this.jobId = ''});

  @override
  State<JobStatusScreen> createState() => _JobStatusScreenState();
}

class _JobStatusScreenState extends State<JobStatusScreen> {
  String status = 'Fetching status...';
  Timer? timer;

  @override
  void initState() {
    super.initState();
    fetchStatus();
    timer = Timer.periodic(const Duration(seconds: 6), (_) => fetchStatus());
  }

  Future<void> fetchStatus() async {
    try {
      final data = await ApiService.getJobStatus(widget.jobId);
      setState(() => status = data['status'] ?? 'Unknown');
    } catch (e) {
      setState(() => status = 'Error: $e');
    }
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Job: ${widget.jobId}'),
        backgroundColor: Colors.deepPurple,
      ),
      body: Center(
        child: Text(
          status,
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
      ),
    );
  }
}
