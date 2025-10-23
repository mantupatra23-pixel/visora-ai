import 'package:flutter/material.dart';
import '../services/api_service.dart';

class AdminControls extends StatefulWidget {
  const AdminControls({super.key});

  @override
  State<AdminControls> createState() => _AdminControlsState();
}

class _AdminControlsState extends State<AdminControls> {
  bool loading = true;
  Map<String, dynamic>? data;
  String error = '';

  @override
  void initState() {
    super.initState();
    fetchAdminData();
  }

  Future<void> fetchAdminData() async {
    try {
      final res = await ApiService.get('/api/admin/config');
      setState(() {
        data = res;
        loading = false;
      });
    } catch (e) {
      setState(() {
        error = e.toString();
        loading = false;
      });
    }
  }

  Future<void> updatePlan(String plan, int price) async {
    try {
      await ApiService.createVideoJob(
        title: "Admin Update",
        script: "Updating pricing",
        options: {"plan": plan, "price": price},
      );
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Pricing updated successfully!')),
      );
      fetchAdminData();
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    if (error.isNotEmpty) {
      return Scaffold(
        body: Center(child: Text('Error: $error')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Visora Admin Controls"),
        backgroundColor: Colors.deepPurple,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text("💰 Subscription Plans",
              style: TextStyle(
                  fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black)),
          const SizedBox(height: 10),
          _planCard("Free", data?['plans']?['free'] ?? 0),
          _planCard("Premium", data?['plans']?['premium'] ?? 0),
          _planCard("Enterprise", data?['plans']?['enterprise'] ?? 0),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: fetchAdminData,
            icon: const Icon(Icons.refresh),
            label: const Text("Reload Data"),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple),
          ),
        ],
      ),
    );
  }

  Widget _planCard(String name, dynamic value) {
    final ctrl = TextEditingController(text: value.toString());
    return Card(
      elevation: 3,
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(name, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            SizedBox(
              width: 100,
              child: TextField(
                controller: ctrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(border: OutlineInputBorder()),
              ),
            ),
            ElevatedButton(
              onPressed: () {
                final price = int.tryParse(ctrl.text) ?? 0;
                updatePlan(name.toLowerCase(), price);
              },
              style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
              child: const Text("Save"),
            ),
          ],
        ),
      ),
    );
  }
}
