import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:lottie/lottie.dart';

class JobStatusScreen extends StatefulWidget {
  final String jobId;
  // backend base url: change if needed
  static const String backendBase = "https://visora-ai-5nqs.onrender.com";

  const JobStatusScreen({
    Key? key,
    required this.jobId,
  }) : super(key: key);

  @override
  State<JobStatusScreen> createState() => _JobStatusScreenState();
}

class _JobStatusScreenState extends State<JobStatusScreen> {
  Timer? _pollTimer;
  Map<String, dynamic>? _data;
  String? _status; // queued / processing / completed / failed
  double _progress = 0.0; // 0.0 - 1.0
  String? _thumbnailUrl;
  String? _resultUrl;
  String? _errorMessage;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _startPolling();
    _fetchStatus(); // immediate fetch
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  void _startPolling() {
    _pollTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      _fetchStatus();
    });
  }

  Future<void> _fetchStatus() async {
    final uri = Uri.parse("${JobStatusScreen.backendBase}/api/job-status/${widget.jobId}");
    try {
      final resp = await http.get(uri).timeout(const Duration(seconds: 8));
      if (resp.statusCode == 200) {
        final Map<String, dynamic> json = jsonDecode(resp.body);
        // expected shape (example):
        // { "status": "processing", "progress": 45, "thumbnail": "...", "result_url": "..." }
        setState(() {
          _data = json;
          _status = (json['status'] as String?)?.toLowerCase() ?? 'unknown';
          final p = json['progress'];
          if (p is num) {
            _progress = (p.toDouble() / 100.0).clamp(0.0, 1.0);
          } else {
            _progress = 0.0;
          }
          _thumbnailUrl = json['thumbnail'] as String?;
          _resultUrl = json['result_url'] as String? ?? json['download_url'] as String?;
          _errorMessage = json['error'] as String?;
          _loading = false;
        });

        if (_status == 'completed' || _status == 'failed') {
          // stop polling after terminal state
          _pollTimer?.cancel();
        }
      } else {
        setState(() {
          _errorMessage = "Server error: ${resp.statusCode}";
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Network error";
        _loading = false;
      });
    }
  }

  Widget _buildTopCard() {
    final theme = Theme.of(context);
    final statusText = (_status ?? 'queued').toUpperCase();
    Color statusColor;
    switch (_status) {
      case 'processing':
        statusColor = Colors.orangeAccent;
        break;
      case 'completed':
        statusColor = Colors.greenAccent;
        break;
      case 'failed':
        statusColor = Colors.redAccent;
        break;
      default:
        statusColor = Colors.blueAccent;
    }

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1B2F),
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.4), blurRadius: 12, offset: const Offset(0, 6)),
        ],
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 28,
            backgroundColor: statusColor.withOpacity(0.2),
            child: Icon(
              _status == 'completed' ? Icons.check_circle : (_status == 'failed' ? Icons.error : Icons.cloud_upload),
              color: statusColor,
              size: 32,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Job ID: ${widget.jobId}",
                  style: theme.textTheme.bodyText1?.copyWith(color: Colors.white70),
                ),
                const SizedBox(height: 6),
                Text(
                  statusText,
                  style: theme.textTheme.headline6?.copyWith(color: Colors.white, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 6),
                LinearProgressIndicator(
                  value: _status == 'completed' ? 1.0 : _progress,
                  minHeight: 6,
                  backgroundColor: Colors.white12,
                  valueColor: AlwaysStoppedAnimation(statusColor),
                ),
                const SizedBox(height: 6),
                Text(
                  _status == 'processing'
                      ? "Processing ${(_progress * 100).toStringAsFixed(0)}%"
                      : (_status == 'completed' ? "Completed" : ( _status == 'failed' ? (_errorMessage ?? 'Failed') : "Queued")),
                  style: theme.textTheme.caption?.copyWith(color: Colors.white60),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPreviewArea() {
    if (_loading) {
      return Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            if (_hasLottieAsset()) Lottie.asset('assets/lottie/loading.json', height: 120),
            const SizedBox(height: 12),
            const Text("Checking job status...", style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }

    if (_errorMessage != null && (_status == 'failed' || _status == null)) {
      return Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Lottie.asset('assets/lottie/done.json', height: 120, animate: false),
            const SizedBox(height: 12),
            Text("Error: $_errorMessage", style: const TextStyle(color: Colors.redAccent)),
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _loading = true;
                  _errorMessage = null;
                });
                _startPolling();
                _fetchStatus();
              },
              child: const Text("Retry"),
            )
          ],
        ),
      );
    }

    if (_status == 'completed') {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_thumbnailUrl != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.network(_thumbnailUrl!, height: 200, fit: BoxFit.cover, loadingBuilder: (ctx, child, prog) {
                  if (prog == null) return child;
                  return SizedBox(
                    height: 200,
                    child: Center(child: CircularProgressIndicator(value: prog.expectedTotalBytes != null ? prog.cumulativeBytesLoaded / (prog.expectedTotalBytes ?? 1) : null)),
                  );
                }),
              ),
              const SizedBox(height: 12),
            ],
            ElevatedButton.icon(
              icon: const Icon(Icons.play_arrow),
              label: const Text("Play Preview"),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurpleAccent),
              onPressed: _resultUrl != null ? () => _openResult(_resultUrl!) : null,
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              icon: const Icon(Icons.download),
              label: const Text("Download / Open File"),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.tealAccent.shade700, foregroundColor: Colors.black),
              onPressed: _resultUrl != null ? () => _openResult(_resultUrl!) : null,
            ),
          ],
        ),
      );
    }

    // queued / processing
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          if (_hasLottieAsset()) Lottie.asset('assets/lottie/loading.json', height: 130),
          const SizedBox(height: 12),
          Text(
            _status == 'processing' ? "Your job is being processed..." : "Job is queued. Waiting to start...",
            style: const TextStyle(color: Colors.white70),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 12),
          Text(
            "Progress: ${(_progress * 100).toStringAsFixed(0)}%",
            style: const TextStyle(color: Colors.white60),
          ),
        ],
      ),
    );
  }

  bool _hasLottieAsset() {
    // if you included Lottie assets earlier, show them
    // otherwise false to avoid runtime error if asset missing
    // User added assets/lottie/loading.json and done.json — we assume present
    return true;
  }

  void _openResult(String url) {
    // open url — here we try to launch using url_launcher if available.
    // Simple fallback: show dialog with link for copy/paste.
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("Open Result"),
        content: Text(url),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close")),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              // try to launch using url_launcher if added in project
              // else user can copy paste
            },
            child: const Text("Copy / Open"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F1020),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0B0B1A),
        elevation: 0,
        title: const Text("Job Status"),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          children: [
            _buildTopCard(),
            Expanded(
              child: SingleChildScrollView(
                child: _buildPreviewArea(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      icon: const Icon(Icons.refresh, color: Colors.white60),
                      label: const Text("Refresh", style: TextStyle(color: Colors.white70)),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: Colors.white12),
                        backgroundColor: const Color(0xFF121221),
                      ),
                      onPressed: () {
                        setState(() {
                          _loading = true;
                        });
                        _fetchStatus();
                      },
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton.icon(
                      icon: const Icon(Icons.home),
                      label: const Text("Back"),
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurpleAccent),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
