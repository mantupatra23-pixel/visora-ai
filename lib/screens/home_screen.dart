import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/api_service.dart';
import 'templates_screen.dart';
import 'assistant_screen.dart';
import 'job_status_screen.dart';
import 'profile_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _ideaCtrl = TextEditingController();
  bool _loading = false;
  List<dynamic> _recentJobs = [];

  @override
  void initState() {
    super.initState();
    _loadRecent();
  }

  Future<void> _loadRecent() async {
    try {
      final res = await ApiService.get('/api/jobs/recent'); // backend endpoint
      setState(() {
        _recentJobs = res is List ? res : [];
      });
    } catch (_) {
      // ignore
    }
  }

  Future<void> _createQuickJob() async {
    final txt = _ideaCtrl.text.trim();
    if (txt.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('कृपया कुछ टाइप करें')));
      return;
    }
    setState(() => _loading = true);
    try {
      final job = await ApiService.createVideoJob(title: 'Quick: ${txt.substring(0, txt.length>30?30:txt.length)}', script: txt);
      final jobId = job['id'] ?? job['_id'] ?? job['jobId'];
      if (jobId != null) {
        Navigator.push(context, MaterialPageRoute(builder: (_) => JobStatusScreen(jobId: jobId)));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    } finally {
      setState(() => _loading = false);
    }
  }

  Widget _featureChip(IconData ic, String title, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0,4))
        ]),
        child: Row(children: [Icon(ic, color: Colors.deepPurple), const SizedBox(width:8), Text(title, style: TextStyle(fontWeight: FontWeight.w600))]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('VISORA', style: GoogleFonts.poppins(fontWeight: FontWeight.w700)),
        backgroundColor: Colors.deepPurple,
        actions: [
          IconButton(onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ProfileScreen())), icon: const Icon(Icons.person))
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadRecent,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Banner
            ClipRRect(borderRadius: BorderRadius.circular(14), child: Image.asset('assets/logo/visora_brand_banner.png', fit: BoxFit.cover)),
            const SizedBox(height:12),
            Text('Quick Create', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w700)),
            const SizedBox(height:8),
            TextField(
              controller: _ideaCtrl,
              minLines: 1,
              maxLines: 4,
              decoration: InputDecoration(
                hintText: 'कुछ लिखो — मैं AI script में बदल दूंगा...',
                filled: true,
                fillColor: Colors.white,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                suffixIcon: IconButton(icon: _loading ? const CircularProgressIndicator() : const Icon(Icons.play_arrow), onPressed: _loading ? null : _createQuickJob),
              ),
            ),
            const SizedBox(height:16),
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Expanded(child: _featureChip(Icons.video_camera_front, 'AI Avatar', (){})),
              const SizedBox(width:8),
              Expanded(child: _featureChip(Icons.movie_filter, 'Templates', () => Navigator.push(context, MaterialPageRoute(builder: (_) => const TemplatesScreen())))),
              const SizedBox(width:8),
              Expanded(child: _featureChip(Icons.chat_bubble_outline, 'Assistant', () => Navigator.push(context, MaterialPageRoute(builder: (_) => const AssistantScreen())))),
            ]),
            const SizedBox(height:20),
            Text('Recent Jobs', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height:8),
            if (_recentJobs.isEmpty)
              Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)), child: const Text('कोई हालिया जॉब नहीं')),
            ..._recentJobs.map((j) {
              final id = j['id'] ?? j['_id'] ?? j['jobId'] ?? '';
              final status = j['status'] ?? 'unknown';
              return ListTile(
                tileColor: Colors.white,
                title: Text(j['title'] ?? 'Untitled'),
                subtitle: Text('Status: $status'),
                trailing: TextButton(child: const Text('Open'), onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => JobStatusScreen(jobId: id)));
                }),
              );
            }).toList(),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}
