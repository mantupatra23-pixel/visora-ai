// lib/main.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(VisoraApp());
}

class VisoraApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Visora - AI Video Studio',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _scriptCtrl = TextEditingController();
  bool _loading = false;
  String _jobId = '';
  String _preview = '';

  // Replace with your actual backend URL (Render / any)
  final String backend = 'https://visora-render.onrender.com';

  Future<void> _startRender() async {
    final script = _scriptCtrl.text.trim();
    if (script.isEmpty) {
      _show('Enter a script or prompt first.');
      return;
    }
    setState(() {
      _loading = true;
      _jobId = '';
      _preview = '';
    });

    try {
      final url = Uri.parse('$backend/generate-video');
      final resp = await http.post(url,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'script': script,
            'voice': 'female_standard',
            'resolution': '720p'
          }));
      final body = jsonDecode(resp.body);
      if (body != null && body['jobId'] != null) {
        setState(() {
          _jobId = body['jobId'];
        });
        _show('Render started. JobId: ${body['jobId']}\nCheck dashboard later.');
      } else if (body != null && body['download_url'] != null) {
        setState(() {
          _preview = body['download_url'];
        });
        _show('Preview ready.');
      } else {
        _show('Failed to start render. Check backend.');
      }
    } catch (e) {
      _show('Error: $e');
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  void _show(String txt) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(txt)));
  }

  Widget _featureTile(String t) {
    return ListTile(
      title: Text(t),
      leading: Icon(Icons.check_circle_outline),
    );
  }

  @override
  Widget build(BuildContext context) {
    final features = [
      'Script → Video',
      'Image → Video (lip-sync)',
      'Audio → Video',
      'Text-to-Speech → Video',
      'AI Avatar Video',
      'Slideshow Maker',
      'Auto Subtitles',
      'Auto SEO Tags',
      'Multiple Resolutions (480p/720p/1080p)',
      'Template Marketplace (placeholder)',
      'Backend rendering on Render (placeholder)',
      // add more textual placeholders if you want...
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text('Visora - AI Video Studio'),
        actions: [
          IconButton(
            icon: Icon(Icons.cloud),
            onPressed: () => _show('Backend: $backend'),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(14),
        child: Column(
          children: [
            Text('Quick Create', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            TextField(
              controller: _scriptCtrl,
              maxLines: 6,
              decoration: InputDecoration(border: OutlineInputBorder(), hintText: 'Enter a prompt or script...'),
            ),
            SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    icon: _loading ? SizedBox(width: 16, height: 16, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : Icon(Icons.play_arrow),
                    label: Text(_loading ? 'Starting...' : 'Generate Video'),
                    onPressed: _loading ? null : _startRender,
                  ),
                ),
              ],
            ),
            SizedBox(height: 12),
            if (_jobId.isNotEmpty) ...[
              ListTile(
                leading: Icon(Icons.timelapse),
                title: Text('Render Job Started'),
                subtitle: Text('Job ID: $_jobId'),
              ),
            ],
            if (_preview.isNotEmpty) ...[
              ListTile(
                leading: Icon(Icons.play_circle_outline),
                title: Text('Preview Ready'),
                subtitle: Text(_preview),
                onTap: () => _show('Copy & open this URL in browser'),
              ),
            ],
            Divider(),
            Text('Features (placeholders)', style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 6),
            ...features.map((f) => _featureTile(f)).toList(),
            SizedBox(height: 20),
            ElevatedButton.icon(
              icon: Icon(Icons.code),
              label: Text('Open GitHub (repo)'),
              onPressed: () => _show('Open your GitHub repo in browser to download APK once built.'),
            ),
            SizedBox(height: 40),
            Text('Note: Heavy features (voice cloning, FFmpeg edits, 3D) run server-side. This app calls backend APIs.'),
          ],
        ),
      ),
    );
  }
}
